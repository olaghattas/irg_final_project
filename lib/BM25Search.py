
import os
from pyserini.search.lucene import LuceneSearcher
from datasets import load_from_disk

def run_bm25(index_dir="pyserini_index", query_config="LitSearch_query", k=50):
    run_filename=f"bm25_top_{k}.run"
    # Resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(project_root, "dataset")
    run_dir = os.path.join(project_root, "run_files")
    os.makedirs(run_dir, exist_ok=True)

    index_path = os.path.join(project_root, index_dir)
    query_path = os.path.join(dataset_dir, query_config)
    run_path = os.path.join(run_dir, run_filename)

    # Load BM25 searcher
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=0.9, b=0.4)

    # Load query dataset
    dataset_query = load_from_disk(query_path)

    with open(run_path, 'w', encoding='utf8') as outfile:
        for ind, qline in enumerate(dataset_query['full']):
            query_text = qline['query']
            hits = searcher.search(query_text, k=k)
            for rank, hit in enumerate(hits, start=1):
                outfile.write(f"{ind} Q0 {hit.docid} {rank} {hit.score:.6f} bm25\n")