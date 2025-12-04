import numpy as np
from tqdm import tqdm
from collections import Counter  
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


class ltc_nnn_scratch:
    def __init__(self, documents):
        self.documents=documents
    
    def ltc(self):
        """
        Compute ltc tf-idf vectors for each document in self.documents.
        Store the result in self.ltc_vectors as a dict {docid: {term: weight, ...}, ...}. 
        """
        
        df = defaultdict(int)

        # tokenize and compute document frequency
        for docid, doc in tqdm(self.documents.items(), 'tokenizing documents...'):
            tokens = set(tokenize(doc))
            for term in tokens:
                df[term] += 1
        N=len(self.documents)
        print(f"Total terms: {len(df)} from {N} documents.")

        # compute idf
        idf = {term: np.log10(N / df_val) for term, df_val in df.items()}

        self.ltc_vectors = {}

        #calculate tf-idf vector for each document
        for docid, doc in tqdm(self.documents.items(), 'calculating tf-idf...'):
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

            self.ltc_vectors[docid] = tfidf 


    def nnn_search(self, query, top_k=5):
        """
        query: str 
        top_k: int
        return: top_k docids and scores 
        """
        q_tokens=tokenize(query)
        q_tf = Counter(q_tokens)


        dists={}
        for docid, doc_vector in self.ltc_vectors.items():
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
