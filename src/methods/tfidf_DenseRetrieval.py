import os
from datasets import load_from_disk
import asyncio

import sys
sys.path.append('.')

from pyserini.search.lucene import LuceneSearcher
from src.classes.query_expansion_LLM import QueryExpansion
import json

from src.classes.dense_retrieval import DenseRetrieval

import torch

### Get LLM Expansion
def get_tfidf(query_config="LitSearch_query"):
    run_filename=f"tfidf_lnc_nnn.run"
    
    # Resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dataset_dir = os.path.join(project_root, "dataset")
    run_dir = os.path.join(project_root, "run_files")
    os.makedirs(run_dir, exist_ok=True)


    query_path = os.path.join(dataset_dir, query_config)
    run_path = os.path.join(run_dir, run_filename)

    ## if run path is already there and not overwrite 
    assert os.path.exists(run_path), f"Path does not exist: {run_path}"
        
    # Load query dataset
    dataset_query = load_from_disk(query_path)
    queries = [item["query"] for item in dataset_query["full"]]

    return run_path, queries

def load_top_docs(run_path, top_k=1000):
    """
    Returns a dictionary mapping query_id -> list of top_k doc_ids.
    """
    top_docs = {}
    with open(run_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            qid = int(parts[0])       # first column: query id
            doc_id = int(parts[2])    # third column: doc id

            if qid not in top_docs:
                top_docs[qid] = []

            if len(top_docs[qid]) < top_k:
                top_docs[qid].append(doc_id)

    return top_docs

def save_dense_run(results_per_query, run_file):
    """
    results_per_query: dict mapping query_id -> list of (doc_id, score)
    """
    with open(run_file, "w", encoding="utf8") as f:
        for qid, results in results_per_query.items():
            for rank, (doc_id, score) in enumerate(results, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} dense\n")
    print(f"Dense retrieval run file saved to: {run_file}")
    
async def main():
    
    run_path, queries = get_tfidf()

    print(run_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    corpus_config= "LitSearch_corpus_clean"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dataset_dir = os.path.join(project_root, "dataset")
    run_dir = os.path.join(project_root, "run_files")
    corpus_path = os.path.join(dataset_dir, corpus_config)
    corpus = load_from_disk(corpus_path)

    dense = DenseRetrieval(corpus["full"], device=device)

    faiss_path = os.path.join(dataset_dir,"dense_index_full.index")
    emb_path = os.path.join(dataset_dir,"dense_embeddings.npz")
    dense.load_index_and_embeddings(faiss_path=faiss_path, emb_path=emb_path)

    tfidf_results = load_top_docs(run_path, top_k=50)
    
    # Dictionary to store all results
    all_dense_results = {}

    # Re-rank using dense retrieval per query
    for qid, query_text in enumerate(queries):
        candidate_doc_ids = tfidf_results.get(qid, [])
        if not candidate_doc_ids:
            print(f"No BM25 results for query {qid}")
            continue

        subset_index = dense.filter_to_specified_ids(candidate_doc_ids)
        results = dense.score_filtered(query_text, subset_index, top_k=50)
        all_dense_results[qid] = results

    # Save as a single run file
    dense_run_file = os.path.join(run_dir, "dense_tfifd_50.run")
    save_dense_run(all_dense_results, dense_run_file)
    

    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())