import argparse
import time

import json

import os, re, time, random, logging, sys, csv, datetime as dt
from pathlib import Path
from typing import Iterable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# download pdf from arxiv while respecting robots.txt crawl-delay

ARXIV_HOST = "export.arxiv.org"     # use export host for batch downloads
ROBOTS_URL = f"https://{ARXIV_HOST}/robots.txt"
USER_AGENT = "IGR-ResearchDownloader/1.0 (+contact: your_email@edu.edu)"

# --- setup logging ---
log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
log_path = log_dir / "arxiv_download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("arxiv-dl")

logger.info(f"------------ Starting new run ------------\n")


CSV_FIELDS = [
    "timestamp_iso", "arxiv_id", "status", "http_status",
    "size_bytes", "size_mb", "total_mb_so_far", "file_path"
]

def csv_write_row(csv_path: Path, row: dict):
    """Append a row to CSV, writing header if file doesn't exist yet."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        # Ensure all fields present
        for k in CSV_FIELDS:
            row.setdefault(k, "")
        w.writerow(row)

def now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def get_crawl_delay(default_delay: float = 15.0) -> float:
    try:
        r = requests.get(ROBOTS_URL, timeout=20, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        text = r.text
        ua_blocks = re.split(r"(?im)^User-agent:\s*", text)
        for block in ua_blocks:
            if block.startswith("*"):
                m = re.search(r"(?im)^Crawl-delay:\s*([0-9]+)", block)
                if m:
                    delay = float(m.group(1))
                    logger.info(f"robots.txt crawl-delay found: {delay:.0f}s")
                    return delay
                break
    except Exception as e:
        logger.warning(f"Could not read robots.txt ({e}); using default {default_delay}s")
    logger.info(f"robots.txt crawl-delay not specified; using default {default_delay}s")
    return default_delay

def build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[500, 502, 504],  # 503/403/429 => denial -> stop (no retry)
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    s.headers.update({"User-Agent": USER_AGENT})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def download_pdfs(
    paper_ids: Iterable[str],
    save_dir: str,
    max_count: int = 1000, 
    summary_csv: str = "logs/arxiv_download_summary.csv",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(summary_csv)

    sess = build_session()
    crawl_delay = get_crawl_delay()  
    downloaded = 0
    total_mb = 0.0

    with tqdm(total=max_count, desc="Downloading PDFs", unit="pdf") as pbar:
        for pid in paper_ids: 
            if downloaded >= max_count:
                logger.info(f"Reached max_count={max_count}. Stopping.")
                break

            pid = pid.strip()
            pdf_url = f"https://{ARXIV_HOST}/pdf/{pid}.pdf"
            out_path = save_dir / f"{pid}.pdf"

            # If already exists, count its size into total and log/CSV as skipped
            if out_path.exists() and out_path.stat().st_size > 0:
                size_mb = out_path.stat().st_size / (1024 * 1024)
                # total_mb += size_mb 
                logger.info(f"Already exists: {out_path.name} ({size_mb:.2f} MB) | Total so far: {total_mb:.2f} MB")
                csv_write_row(csv_path, {
                    "timestamp_iso": now_iso(),
                    "arxiv_id": pid,
                    "status": "skipped_exists",
                    "http_status": 200,
                    "size_bytes": out_path.stat().st_size,
                    "size_mb": f"{size_mb:.4f}",
                    "total_mb_so_far": f"{total_mb:.4f}",
                    "file_path": str(out_path.resolve()),
                })
                # downloaded += 1
                # time.sleep(crawl_delay * random.uniform(0.9, 1.1))
                continue

            try:
                with sess.get(pdf_url, timeout=60, stream=True) as r:
                    status = r.status_code

                    # Stop on denial
                    if status in (403, 429, 503):
                        logger.error(f"Server denied (status {status}) on {pdf_url}. Stopping.")
                        csv_write_row(csv_path, {
                            "timestamp_iso": now_iso(),
                            "arxiv_id": pid,
                            "status": "denied_stop",
                            "http_status": status,
                            "size_bytes": 0,
                            "size_mb": "0.0000",
                            "total_mb_so_far": f"{total_mb:.4f}",
                            "file_path": str(out_path.resolve()),
                        })
                        break

                    if status != 200:
                        logger.warning(f"Failed {pid}: HTTP {status}")
                        csv_write_row(csv_path, {
                            "timestamp_iso": now_iso(),
                            "arxiv_id": pid,
                            "status": "failed",
                            "http_status": status,
                            "size_bytes": 0,
                            "size_mb": "0.0000",
                            "total_mb_so_far": f"{total_mb:.4f}",
                            "file_path": str(out_path.resolve()),
                        })
                    else:
                        tmp_path = out_path.with_suffix(".part")
                        with open(tmp_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1024 * 64):
                                if chunk:
                                    f.write(chunk)

                        if tmp_path.stat().st_size == 0:
                            logger.warning(f"Empty file for {pid}; removing.")
                            tmp_path.unlink(missing_ok=True)
                            csv_write_row(csv_path, {
                                "timestamp_iso": now_iso(),
                                "arxiv_id": pid,
                                "status": "empty_file",
                                "http_status": status,
                                "size_bytes": 0,
                                "size_mb": "0.0000",
                                "total_mb_so_far": f"{total_mb:.4f}",
                                "file_path": str(out_path.resolve()),
                            })
                        else:
                            tmp_path.replace(out_path)
                            size_b = out_path.stat().st_size
                            size_mb = size_b / (1024 * 1024)
                            total_mb += size_mb
                            logger.info(f"OK {pid} -> {out_path.name} ({size_mb:.2f} MB) | Total so far: {total_mb:.2f} MB")
                            csv_write_row(csv_path, {
                                "timestamp_iso": now_iso(),
                                "arxiv_id": pid,
                                "status": "ok",
                                "http_status": status,
                                "size_bytes": size_b,
                                "size_mb": f"{size_mb:.4f}",
                                "total_mb_so_far": f"{total_mb:.4f}",
                                "file_path": str(out_path.resolve()),
                            })
                            downloaded += 1
                            pbar.update(1)   # move progress bar forward

            except KeyboardInterrupt:
                logger.info("Interrupted by user. Exiting cleanly.")
                break
            except requests.RequestException as e:
                logger.warning(f"Network error on {pid}: {e}")
                csv_write_row(csv_path, {
                    "timestamp_iso": now_iso(),
                    "arxiv_id": pid,
                    "status": "network_error",
                    "http_status": "",
                    "size_bytes": 0,
                    "size_mb": "0.0000",
                    "total_mb_so_far": f"{total_mb:.4f}",
                    "file_path": str(out_path.resolve()),
                })

            time.sleep(crawl_delay * random.uniform(0.9, 1.1))

    logger.info(f"Finished. Downloaded {downloaded} PDFs, {total_mb:.2f} MB total. Log at: {log_path} | CSV: {csv_path}")



def main(id_files_json, category, output_dir, max_count):
     
    with open(id_files_json, 'r') as f:
        cats_ids = json.load(f)
    
    print(f"Total categories in {id_files_json}: {len(cats_ids)}")
    print(f"Category to download: {category} with {max_count} / {len(cats_ids[category])} papers")

    paper_ids = sorted(cats_ids[category])
 
    download_pdfs(
        paper_ids,
        save_dir=f"{output_dir}/{category}",
        max_count=max_count, 
        summary_csv="logs/arxiv_download_summary.csv",
    )





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ArXiv PDFs politely.")
    parser.add_argument("--id_files_json", type=str, default='cs_categories_paper_ids.json', help="Path to the cat:id list json file.")
    parser.add_argument("--output_dir", type=str, default="arxiv_papers", help="Directory to save downloaded PDFs.")
    parser.add_argument("--category", type=str, default="cs.RO", help="Category to download (e.g., cs.RO)")
    parser.add_argument("--max_count", type=int, default=10, help="Maximum number of PDFs to download.")
    args = parser.parse_args()
    main(args.id_files_json, args.category, args.output_dir, args.max_count)

