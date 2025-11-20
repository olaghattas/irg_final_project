import sys
sys.path.append('.')

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from index import Index
from query import Query

class ColBert(Index):

    def __init__(self, dir):
        self.dir = dir
        np_items = np.load(f"{dir}/embeddings.npz")
        self.embeddings = np_items['embeddings']
        self.doc_ids = np_items['doc_ids']
        self.sentences = np_items['sentences']
        self.index = faiss.read_index(f"{dir}/index.faiss")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def search(self, query_embedding, doc_embedding_type, k):
        dist, inds = self.index.search(query_embedding,k)
        top_doc_ids = np.unique(self.doc_ids[inds.flatten()])
        doc_sims = []
        for d_id in top_doc_ids:
            doc_sims.append((str(d_id),float(max([self.model.similarity(query_s,embedding)  for query_s in query_embedding for embedding in self.embeddings[self.doc_ids==d_id]])))) 
        return sorted(doc_sims, key=lambda x: x[1], reverse=True)[0:k]
    
    def get_contents(self, d_ids):
        return [''.join(self.sentences[self.doc_ids == d_id]) for d_id in d_ids]
    

def main():
    index = ColBert("indexes/colbert_index")
    query = Query.model_validate_json("{ \"contents\" : \"Robots are going to 3D imaging. make better motions\" }")
    ranking = index.search(query.get_colbert(index.model),'',10)
    print(ranking)
    texts = index.get_contents([rank[0] for rank in ranking])
    for (i,text) in enumerate(texts):
        print(f"---------------------------------RESULT {i}:--------------------------------------")
        print(text)
    return

if __name__ == "__main__":
    main()


