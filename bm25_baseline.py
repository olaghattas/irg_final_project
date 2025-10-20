from pyserini.search.lucene import LuceneSearcher
from datasets import load_from_disk

index_path = "pyserini_index"
qrel_path = "litsearch.qrel"
query_dataset = "LitSearch_query"
run_path = "bm25.run"

lucene_bm25_searcher = LuceneSearcher(index_path)
lucene_bm25_searcher.set_bm25(k1=0.9, b=0.4) 

dataset_query = load_from_disk(query_dataset)
with open(run_path, 'w', encoding='utf8') as outfile:
    for (ind,qline) in enumerate(dataset_query['full']):
        print(qline['query'])
        hits = lucene_bm25_searcher.search(qline['query'], k=50)
        for rank, hit in enumerate(hits, start=1):
            outfile.write(f"{ind} Q0 {hit.docid} {rank} {hit.score:.6f} bm25\n")

