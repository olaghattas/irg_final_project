
"""
evaluate.py — Compute metrics for many runs vs qrels
- Supports metrics: ndcg@50 (default), p@k, p@R, ap, map
- Saves per-query CSV and summary CSV
- Computes mean, stderr per system

python3 evaluate.py --metric ndcg@50
or 
python3 evaluate.py --metric map
"""
import  math
from pathlib import Path
from metrics import read_qrels, read_run, per_query_metric
import csv

qrel_file = "evaluation/litsearch.qrel"
# default_metric = "ndcg@10" ## Available metrics: ndcg@K | p@K | p@R | ap | map
output_path = "evaluation/results"
run_files = ["run_files/dense_expanded_100.run"]

def stderr(x):
    n = len(x)
    if n <= 1: 
        return 0.0
    mean = sum(x)/n
    var = sum((xi-mean)**2 for xi in x)/(n-1)
    return math.sqrt(var/n)

def main():

    metrics =  ["ndcg@50", "p@50", "ap", "map"]
  
    for metric in metrics:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        qrels = read_qrels(qrel_file, graded_ok=True)

        summary_rows = []
        for run_path in run_files:
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
        with open(out / f"summary_{name}_{metric}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", f"mean_{metric}", "stderr"])
            for row in summary_rows:
                w.writerow(row)

if __name__ == "__main__":
    main()
