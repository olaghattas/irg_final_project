import argparse
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
from collections import Counter  
from datasets import load_from_disk 
from collections import Counter, defaultdict 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk 

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

from sentence_transformers import CrossEncoder
from typing import List, Tuple

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(CROSS_ENCODER_MODEL)


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

def load_corpus_query(corpus, query):
    data = load_from_disk(corpus)
    documents={}    
    for item in tqdm(data["full"], desc="Processing documents"):
        doc_id = str(item["corpusid"])
        contents = (item["title"] or "") + " " + (item["abstract"] or "")
        documents[doc_id] = contents
    print('Total documents: ', len(documents)) 


    dataset_query = load_from_disk("dataset/LitSearch_query") 
    requests={}
    for i in range(len(dataset_query['full'])):
        query_text = dataset_query['full'][i]['query']
        requests[i] = query_text.lower()
    print(f"Total queries: {len(requests)}")
    return documents, requests

def save_run_file(all_query_results, output_path, method_name):
    """
    Save the query results to a run file.
    all_query_results: dict, {qid: [(doc_id, score), ...], ...}
    output_path: str, path to save the run file
    method_name: str, "ltc_nnn"
    """
    with open(output_path, "w", encoding="utf-8") as out_f:
        for qid in all_query_results:
            results = all_query_results[qid]
            
            rank = 0
            for doc_id, score in results:
                rank += 1
                line = f"{qid} Q0 {doc_id} {rank} {score:.6f} {method_name}\n"
                # print(line.strip())
                out_f.write(line)


def main(args):
    ranked = read_run(args.run_file) 
    # print('candidates: ', ranked)
    print('loading data...')
    documents, requests = load_corpus_query(args.corpus, args.query)
    docids=[]
    corpus=[]
    for doc_id, doc in documents.items():
        docids.append(doc_id)
        corpus.append(doc)

    print('reranking with cross-encoder...')
    results={}
    for qid in tqdm(requests):
        query = requests[qid]
        ranking=ranked[str(qid)]
        top_docs=ranking

        pairs=[(query, documents[index]) for index in ranking]
        scores = reranker.predict(pairs, convert_to_numpy=True)
        order = np.argsort(-scores)  # descending

        reranked_top_docs=[top_docs[i]  for i in order] 
        results[qid]= zip(reranked_top_docs, np.sort(-scores))

    method_name = "ce"
    output_file = f"run_files/{method_name}_topk_{args.topk}.run"
    print(f"saving to {output_file}")
    save_run_file(results, output_file, method_name)
    print('---all done---')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_file", type=str, required=True, help="Path to the initial run file")  
    parser.add_argument("--corpus", type=str, required=True, help="Path to the corpus dataset") 
    parser.add_argument("--query", type=str, required=True, help="Path to the query dataset") 
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()
    main(args)

# cd irg_final_project
# python3 src/methods/run_ce.py \
#     --run_file run_files/ltc_nnn_scratch_topk_1000.run \
#     --corpus dataset/LitSearch_corpus_clean \
#     --query dataset/LitSearch_query \
#     --topk 50

# python3 src/methods/run_ce.py \
#     --run_file run_files/bm25_top_1000.run \
#     --corpus dataset/LitSearch_corpus_clean \
#     --query dataset/LitSearch_query \
#     --topk 50

# cd evaluation
# python evaluate.py --qrels litsearch.qrel --runs /home/ns1254/irg_final_project/run_files/ce_topk_50.run --metric map --output results
# python evaluate.py --qrels litsearch.qrel --runs /home/ns1254/irg_final_project/run_files/ce_topk_50.run --metric ndcg@50 --output results

# ce_topk_50: mean=0.490232, stderr=0.016261
