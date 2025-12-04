import argparse
from tqdm import tqdm
from datasets import load_from_disk 

import sys 
import os

current_file = os.path.abspath(__file__)
methods_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(methods_dir)
sys.path.append(src_dir) 

from classes.ltc_nnn_scratch import ltc_nnn_scratch

def load_corpus_query(corpus, query):
    """ 
    Load the corpus and query datasets from disk.
    corpus: str, path to corpus dataset
    query: str, path to query dataset
    return: documents dict and requests dict
    """
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
    top_k = args.topk
    version = "v1"
    method_name = "ltc_nnn_scratch"

    print('loading data...')
    documents, requests = load_corpus_query(args.corpus, args.query)
    docids=[]
    corpus=[]
    for doc_id, doc in documents.items():
        docids.append(doc_id)
        corpus.append(doc)

    print('training tf-idf')
    ltc_nnn_model = ltc_nnn_scratch(documents)
    ltc_nnn_model.ltc() 

    print('evaluating queries...')
    all_query_results={}
    for qid, query in tqdm(requests.items(), desc="Processing queries"):
        top_idx, top_score= ltc_nnn_model.nnn_search(query, top_k=top_k) 
        all_query_results[qid] = list(zip(top_idx, top_score))

    print('all queries processed.')
    


    output_file = f"run_files/{method_name}_topk_{top_k}.run"
    print(f"saving to {output_file}")
    save_run_file(all_query_results, output_file, method_name)
    print('---all done---')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, help="Path to the corpus dataset") 
    parser.add_argument("--query", type=str, required=True, help="Path to the query dataset") 
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()
    main(args)

# cd irg_final_project
# python3 src/methods/run_scratch.py \
#     --corpus dataset/LitSearch_corpus_clean \
#     --query dataset/LitSearch_query \
#     --topk 50
# python evaluate.py --qrels litsearch.qrel --runs /home/ns1254/irg_final_project/run_files/ltc_nnn_scratch_topk_50.run --metric map --output results
# python evaluate.py --qrels litsearch.qrel --runs /home/ns1254/irg_final_project/run_files/ltc_nnn_scratch_topk_50.run --metric ndcg@50 --output results

