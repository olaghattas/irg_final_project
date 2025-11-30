
"""
evaluate.py — Compute metrics for many runs vs qrels.
- Supports metrics: ndcg@K (default ndcg@20), p@K, p@R, ap, map
- Saves per-query CSV and summary CSV
- Computes mean, stderr per system

Example:
python evaluation/evaluate.py \\
  --qrels evaluation/litsearch.qrel \\
  --metric ndcg@50 \\
  --output evaluation/results \\
  --runs run_files/colbert.run run_files/another.run
"""
import argparse, csv, math
from pathlib import Path
from metrics import read_qrels, read_run, per_query_metric


DEFAULT_QRELS = "evaluation/litsearch.qrel"
DEFAULT_METRIC = "ndcg@20"  # ndcg@K | p@K | p@R | ap | map
DEFAULT_OUTPUT = "evaluation/results"
DEFAULT_RUNS = ["run_files/example.run"]

def stderr(x):
    n = len(x)
    if n <= 1: 
        return 0.0
    mean = sum(x)/n
    var = sum((xi-mean)**2 for xi in x)/(n-1)
    return math.sqrt(var/n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default=DEFAULT_QRELS, help="Qrel file path")
    ap.add_argument("--metric", default=DEFAULT_METRIC, help="ndcg@K | p@K | p@R | ap | map")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Directory to write results")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="One or more TREC run files")
    args = ap.parse_args()
    
    metric = args.metric

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    qrels = read_qrels(args.qrels, graded_ok=True)

    summary_rows = []
    for run_path in args.runs:
        run = read_run(run_path)
        mean, per_q = per_query_metric(run, qrels, metric=metric)
        # write per-query
        name = Path(run_path).stem
        perq_csv = out / f"perq_{name}_{metric}.csv"
        with open(perq_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qid", metric])
            for qid, val in sorted(per_q.items()):
                w.writerow([qid, f"{val:.6f}"])
        se = stderr(list(per_q.values()))
        summary_rows.append([name, f"{mean:.6f}", f"{se:.6f}"])

    # write summary
    summary_name = out / f"summary_{metric}.csv"
    with open(summary_name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", f"mean_{metric}", "stderr"])
        for row in summary_rows:
            w.writerow(row)
    # Echo summary to stdout for quick visibility
    print(f"Summary ({metric})")
    for name, mean, se in summary_rows:
        print(f"- {name}: mean={mean}, stderr={se}")
    print(f"Summary file written to: {summary_name}")

if __name__ == "__main__":
    main()
