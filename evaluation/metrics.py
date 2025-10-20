
"""
metrics.py — Core IR evaluation metrics
Implements: Precision@k, Precision@R, AP, MAP, DCG/NDCG@k
Works with TREC-formatted qrels and run files.
"""
from collections import defaultdict
from math import log2, sqrt

# --------- I/O helpers ---------
def read_qrels(path, graded_ok=True):
    """
    TREC qrels: qid <unused> docid rel
    Returns: dict[qid][docid] = rel (int)
    If graded_ok=False, rel>0 is converted to 1, else 0.
    """
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"): 
                continue
            parts = line.split()
            if len(parts) < 4: 
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            if not graded_ok:
                rel = 1 if rel > 0 else 0
            qrels[qid][docid] = rel
    return qrels

def read_run(path, max_k=None):
    """
    TREC run: qid Q0 docid rank score tag
    Returns: dict[qid] = [docid1, docid2, ...] sorted by rank (ascending)
    """
    per_q = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            qid, _, docid, rank, score, tag = line.split()[:6]
            per_q[qid].append((int(rank), docid))
    ranked = {}
    for qid, pairs in per_q.items():
        pairs.sort(key=lambda x: x[0])
        docs = [d for _, d in pairs]
        ranked[qid] = docs if max_k is None else docs[:max_k]
    return ranked

# --------- Metrics ---------
def precision_at_k(run_docs, qrels_for_q, k):
    if k <= 0:
        return 0.0
    top = run_docs[:k]
    rels = sum(1 for d in top if qrels_for_q.get(d, 0) > 0)
    return rels / k

def precision_at_r(run_docs, qrels_for_q):
    R = sum(1 for r in qrels_for_q.values() if r > 0)
    if R == 0:
        return 0.0
    top = run_docs[:R]
    rels = sum(1 for d in top if qrels_for_q.get(d, 0) > 0)
    return rels / R

def average_precision(run_docs, qrels_for_q):
    """Binary AP. If no relevant docs exist, returns 0."""
    R = sum(1 for r in qrels_for_q.values() if r > 0)
    if R == 0:
        return 0.0
    ap_sum, rel_so_far = 0.0, 0
    for i, d in enumerate(run_docs, start=1):
        if qrels_for_q.get(d, 0) > 0:
            rel_so_far += 1
            ap_sum += rel_so_far / i
    return ap_sum / R

def map_score(run, qrels):
    """Mean of per-query AP over intersection of queries in run and qrels."""
    qs = sorted(set(run.keys()) & set(qrels.keys()))
    if not qs:
        return 0.0, {}
    per_q = {}
    for q in qs:
        per_q[q] = average_precision(run[q], qrels[q])
    mean = sum(per_q.values()) / len(per_q)
    return mean, per_q

def dcg_at_k(run_docs, qrels_for_q, k):
    """Graded DCG@k with gain = rel, discount = log2(1+i)."""
    s = 0.0
    for i, d in enumerate(run_docs[:k], start=1):
        rel = qrels_for_q.get(d, 0)
        if rel > 0:
            s += (2**rel - 1) / log2(1 + i)
    return s

def ndcg_at_k(run_docs, qrels_for_q, k):
    """Graded NDCG@k with gain = 2^rel -1."""
    dcg = dcg_at_k(run_docs, qrels_for_q, k)
    if dcg == 0.0:
        # could still be nonzero ideal; check ideal
        ideal_rels = sorted(qrels_for_q.values(), reverse=True)
        ideal = 0.0
        for i, rel in enumerate(ideal_rels[:k], start=1):
            if rel > 0:
                ideal += (2**rel - 1) / log2(1 + i)
        return 0.0 if ideal == 0.0 else dcg / ideal
    # compute ideal
    ideal_rels = sorted(qrels_for_q.values(), reverse=True)
    ideal = 0.0
    for i, rel in enumerate(ideal_rels[:k], start=1):
        if rel > 0:
            ideal += (2**rel - 1) / log2(1 + i)
    return 0.0 if ideal == 0.0 else dcg / ideal

def per_query_metric(run, qrels, metric="ndcg@20"):
    """
    Returns (mean, per_query_dict). Supports: ndcg@K, p@K, p@R, ap
    """
    qs = sorted(set(run.keys()) & set(qrels.keys()))
    per_q = {}
    if metric.lower().startswith("ndcg@"):
        K = int(metric.split("@")[1])
        for q in qs:
            per_q[q] = ndcg_at_k(run[q], qrels[q], K)
    elif metric.lower().startswith("p@"):
        val = metric.split("@")[1].lower()
        if val == "r":
            for q in qs:
                per_q[q] = precision_at_r(run[q], qrels[q])
        else:
            K = int(val)
            for q in qs:
                per_q[q] = precision_at_k(run[q], qrels[q], K)
    elif metric.lower() == "ap":
        for q in qs:
            per_q[q] = average_precision(run[q], qrels[q])
    elif metric.lower() == "map":
        m, per_q = map_score(run, qrels)
        return m, per_q
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    mean = sum(per_q.values()) / len(per_q) if per_q else 0.0
    return mean, per_q
