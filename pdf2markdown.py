#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Tuple

# Import here so worker processes fail fast if the lib is missing
import pymupdf4llm  # pip install pymupdf4llm

@dataclass(frozen=True)
class Task:
    in_path: Path
    out_path: Path

def discover_pdfs(input_dir: Path, recursive: bool, pattern: str) -> Iterable[Path]:
    glob_pat = f"**/{pattern}" if recursive else pattern
    # Case-insensitive match by normalizing suffix
    for p in input_dir.glob(glob_pat):
        if p.is_file() and p.suffix.lower() == ".pdf":
            yield p

def make_tasks(input_dir: Path, output_dir: Path, pdfs: Iterable[Path]) -> Iterable[Task]:
    for pdf in pdfs:
        rel = pdf.relative_to(input_dir)
        out = (output_dir / rel).with_suffix(".md")
        yield Task(pdf, out)

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def convert_one(task: Task, overwrite: bool) -> Tuple[str, str, float, int]:
    """
    Returns: (status, message, seconds, bytes_written)
    status ∈ {"ok","skip","err"}
    """
    t0 = time.perf_counter()
    try:
        if task.out_path.exists() and not overwrite:
            return ("skip", str(task.out_path), 0.0, 0)
        md_text = pymupdf4llm.to_markdown(str(task.in_path))
        ensure_parent(task.out_path)
        data = md_text.encode("utf-8")
        task.out_path.write_bytes(data)
        dt = time.perf_counter() - t0
        return ("ok", str(task.out_path), dt, len(data))
    except Exception as e:
        dt = time.perf_counter() - t0
        return ("err", f"{task.in_path}: {e}", dt, 0)

def human_mb(nbytes: int) -> str:
    return f"{nbytes/1_000_000:.2f} MB"

def main(args):
    in_dir: Path = args.input_dir.resolve()
    out_dir: Path = args.output_dir.resolve()
    if args.log_dir:
        log_dir = args.log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        args.log_csv = log_dir / "pdf2md_log.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover tasks
    pdfs = list(discover_pdfs(in_dir, args.recursive, args.pattern))
    pdfs.sort()  # stable order
    if args.limit:
        pdfs = pdfs[: args.limit]

    tasks = list(make_tasks(in_dir, out_dir, pdfs))

    total = len(tasks)
    print(f"Found {total} PDF file(s).")
    if total == 0:
        return

    # Pre-scan for skips when overwrite=False to get accurate tqdm total of work
    if not args.overwrite:
        will_do = [t for t in tasks if not t.out_path.exists()]
    else:
        will_do = tasks

    print(f"Planned: {len(will_do)} to convert, {total - len(will_do)} already done.")

    # CSV logging setup
    writer = None
    fcsv = None
    if args.log_csv:
        ensure_parent(args.log_csv)
        new_file = not args.log_csv.exists()
        fcsv = args.log_csv.open("a", newline="", encoding="utf-8")
        writer = csv.writer(fcsv)
        if new_file:
            writer.writerow(["status", "input_pdf", "output_md_or_error", "seconds", "bytes"])

    # Early exit if nothing to do
    if not will_do:
        if fcsv:
            fcsv.close()
        return

    # Use ProcessPool for CPU-bound work (MuPDF parsing)
    convert = partial(convert_one, overwrite=args.overwrite)

    # Defer import to avoid Windows spawn pitfalls in workers (already at top; safe)
    # Run pool
    ok_count = skip_count = err_count = 0
    bytes_total = 0
    failed_files = []
    try:
        from tqdm import tqdm as _tqdm
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(convert, t): t for t in will_do}
            for fut in _tqdm(as_completed(futures), total=len(will_do), unit="pdf", desc="Converting"):
                status, msg, secs, nbytes = fut.result()
                t = futures[fut]

                if status == "ok":
                    ok_count += 1
                    bytes_total += nbytes
                elif status == "skip":
                    skip_count += 1
                else:
                    err_count += 1
                    failed_files.append(str(t.in_path))
                    print(f"[ERR] {msg}", file=sys.stderr)

                if writer:
                    input_pdf = str(t.in_path)
                    output_or_err = msg
                    writer.writerow([status, input_pdf, output_or_err, f"{secs:.4f}", nbytes])
    except KeyboardInterrupt:
        print("\nInterrupted by user. Summarizing partial results…", file=sys.stderr)
    finally:
        if fcsv:
            fcsv.close()

        # Save failed files list
        if failed_files:
            fail_log = log_dir / "failed_files.txt"
            with open(fail_log, "a", encoding="utf-8") as f:
                for pdf in failed_files:
                    f.write(pdf + "\n")
            print(f"\nSaved list of failed files: {fail_log}")

    # Summary
    done = ok_count + skip_count + err_count
    print("\nSummary")
    print(f"  Converted: {ok_count}")
    print(f"  Skipped:   {skip_count}")
    print(f"  Errors:    {err_count}")
    print(f"  Wrote:     {human_mb(bytes_total)} total")
    print(f"  Processed: {done} of {total} total")
    print('-------------------------------------')
    

if __name__ == "__main__":
    # On some platforms, it's safer to use 'spawn' for heavy libraries; uncomment if needed:
    # import multiprocessing as mp
    # mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser(description="Convert PDFs to Markdown (fast, parallel, resumable).")
    ap.add_argument("input_dir", type=Path, help="Directory containing PDFs")
    ap.add_argument("output_dir", type=Path, help="Directory to write Markdown")
    ap.add_argument("--recursive", action="store_true", dest="recursive", help="Recurse into subdirectories")
    ap.add_argument("--no-recursive", action="store_false", dest="recursive", help="Disable recursion")
    ap.set_defaults(recursive=True)
    ap.add_argument("--pattern", default="*.pdf", help="Filename pattern (default: *.pdf)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .md files (default: skip)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of PDFs to process")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes")
    ap.add_argument("--log-dir", type=Path, default=None, help="Path to write log files")
    args = ap.parse_args()
    main(args)


# Example usage:
# python pdf2markdown.py arxiv_papers/cs.RO arxiv_papers/cs.RO_md
# python pdf2markdown.py arxiv_papers/cs.RO arxiv_papers/cs.RO_md --workers 16 --overwrite --log-dir logs_pdf2markdown





