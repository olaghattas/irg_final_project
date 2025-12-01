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

ps=PorterStemmer()
def tokenize(text):
    """nltk tokenizer with stemming."""
    tokens = word_tokenize(text)
    tokens = [ps.stem(token) for token in tokens]
    return tokens 


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


def ltc(documents):
    df = defaultdict(int)

    for docid, doc in tqdm(documents.items(), 'tokenizing documents...'):
        tokens = set(tokenize(doc))
        for term in tokens:
            df[term] += 1
    N=len(documents)
    print(f"Total terms: {len(df)} from {N} documents.")

    idf = {term: np.log10(N / df_val) for term, df_val in df.items()}

    ltc_vectors = {}

    #calculate tf-idf vector for each document
    for docid, doc in tqdm(documents.items(), 'calculating tf-idf...'):
        tokens = tokenize(doc)
        tf_raw = Counter(tokens)

        # l and t weighting
        tfidf = {}
        for term, freq in tf_raw.items():
            tf = 1 + np.log10(freq)    
            tfidf[term] = tf * idf[term]  #lt

        # c normalization
        norm = np.sqrt(sum(v ** 2 for v in tfidf.values())) 
        if norm > 0:
            for term in tfidf:
                tfidf[term] /= norm

        ltc_vectors[docid] = tfidf
    return ltc_vectors


def nnn_search(query, ltc_vectors, top_k=5):
    q_tokens=tokenize(query)
    q_tf = Counter(q_tokens)


    dists={}
    for docid, doc_vector in ltc_vectors.items():
        # nnn → just raw term frequency, no idf, no normalization
        nnn = 0
        for term, freq in q_tf.items():
            if term in doc_vector:
                nnn += freq * doc_vector[term]
        dists[docid] = nnn

    sorted_dists = sorted(dists.items(), key=lambda x: x[1], reverse=True)        #todo: find indices
    tops=sorted_dists[:top_k] 
    top_idx=[docid for docid, _ in tops]
    top_scores=[score for _, score in tops]
    return top_idx, top_scores



def main(args):
    top_k = args.topk
    version = args.version 
    method_name = args.method 

    print('loading data...')
    documents, requests = load_corpus_query(args.corpus, args.query)
    docids=[]
    corpus=[]
    for doc_id, doc in documents.items():
        docids.append(doc_id)
        corpus.append(doc)

    print('training tf-idf')
    ltc_vectors = ltc(documents)

    print('evaluating queries...')
    all_query_results={}
    for qid, query in tqdm(requests.items(), desc="Processing queries"):
        q_tokens=tokenize(query)
        q_tf = Counter(q_tokens)


        dists={}
        for docid, doc_vector in ltc_vectors.items():
            # nnn → just raw term frequency, no idf, no normalization
            nnn = 0
            for term, freq in q_tf.items():
                if term in doc_vector:
                    nnn += freq * doc_vector[term]
            dists[docid] = nnn

        sorted_dists = sorted(dists.items(), key=lambda x: x[1], reverse=True)
        all_query_results[qid] = sorted_dists[:top_k]  

    print('all queries processed.')
    


    output_file = f"{method_name}_v_{version}.run"
    print(f"saving to {output_file}")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for qid in all_query_results:
            results = all_query_results[qid]
            
            rank = 0
            for doc_id, score in results:
                rank += 1
                line = f"{qid} Q0 {doc_id} {rank} {score:.6f} {method_name}\n"
                # print(line.strip())
                out_f.write(line)
            
    print(f"Results written to {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="dataset/LitSearch_corpus_clean") 
    parser.add_argument("--query", type=str, default="dataset/LitSearch_query") 
    parser.add_argument("--method", type=str, default="ltc_nnn" )
    parser.add_argument("--version", type=str, default="stage1") 
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()
    main(args)

#python ns_ltc_nnn.py
