import sys
sys.path.append('.')

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.classes.index import Index
from src.classes.query import Query

from sentence_transformers import SentenceTransformer
import faiss
import os
import torch
from datasets import load_from_disk

class DenseRetrieval(Index):
    def __init__(self, corpus_dataset, model_name="all-MiniLM-L6-v2", device=None):
        # Load the pre-trained model
        self.model = SentenceTransformer(model_name, device=device)
        self.corpus_dataset = corpus_dataset
        
        self.rep = None
        self.index = None
        self.dim = None

        self.corpus_dataset = corpus_dataset 
        self.doc_ids = [doc["corpusid"] for doc in corpus_dataset]
        
    def generate_embeddings(self, faiss_path = None, emb_path=None):
        
        # Combine title + abstract per document
        corpus_texts = [
            doc["title"] + ". " + doc["abstract"] for doc in self.corpus_dataset
        ]
        
        self.rep = self.model.encode(corpus_texts, batch_size=64, convert_to_tensor=True, show_progress_bar=True)

        ## Using FAISS for saving
        corpus_np = self.rep.cpu().numpy().astype('float32')
        
        # Dimension of  embedding
        self.dim = corpus_np.shape[1]
        
        ## Available Indexes
        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        ## Exact search for inner product
        self.index = faiss.IndexFlatIP(self.dim)  
        ## variant Exact Search for L2
        # self.index = faiss.IndexFlatL2(self.dim)  
        
        faiss.normalize_L2(corpus_np) # normalize for cosine similarity
        self.index.add(corpus_np)
        
        print(f"Total sentences indexed: {self.index.ntotal}")

        # Save FAISS index if path provided
        if faiss_path:
            faiss.write_index(self.index, faiss_path)
            print(f"Saved FAISS index to {faiss_path}")
        if emb_path:
            np.savez(emb_path, embeddings=corpus_np, doc_ids=np.array(self.doc_ids))
            print(f"Saved Embedding to {emb_path}")
        
    def get_embeddings(self, faiss_path=None, emb_path=None):
        if faiss_path and os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
            print(f"Loaded FAISS index from {faiss_path}")
            self.dim = self.index.d
            print("self.dim ", self.dim)
        else:
            print("Generating embeddings and FAISS index...")
            self.generate_embeddings(faiss_path=faiss_path, emb_path=emb_path)
    
    def load_index_and_embeddings(self, faiss_path=None, emb_path=None):
        self.get_embeddings(faiss_path, emb_path)
        
        ## Used to create a subset faiss
        if emb_path and os.path.exists(emb_path):
            data = np.load(emb_path)
            self.rep = data['embeddings']
            self.doc_ids = data['doc_ids'].tolist()
            print(f"Loaded embeddings and doc_ids from {emb_path}")
        else:
            print("Embeddings file not found; please generate embeddings first.")
    
    def filter_to_specified_ids(self, candidate_doc_ids):
        candidate_doc_ids = np.array(candidate_doc_ids)
        mask = np.isin(self.doc_ids, candidate_doc_ids)

        vectors_to_add = self.rep[mask]
        # convert to NumPy if it's a PyTorch tensor
        try:
            import torch
            if isinstance(vectors_to_add, torch.Tensor):
                vectors_to_add = vectors_to_add.cpu().numpy()
        except ImportError:
            pass

        vectors_to_add = vectors_to_add.astype('float32')
        ids_to_add = np.array(self.doc_ids)[mask]

        ## not necessary since the faiss index stores the normalized data
        faiss.normalize_L2(vectors_to_add)

        subset_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        subset_index.add_with_ids(vectors_to_add, ids_to_add)

        print(f"Subset index created with {subset_index.ntotal} vectors")
        return subset_index


    def score_filtered(self, query_text, subset_index, top_k=10):
        query_emb = self.model.encode([query_text], convert_to_tensor=True)
        
        if hasattr(query_emb, "cpu"):
            query_np = query_emb.cpu().numpy().astype("float32")
        else:
            query_np = query_emb.astype("float32")
        
        faiss.normalize_L2(query_np)

        distances, indices = subset_index.search(query_np, top_k)

        # Indices from IndexIDMap are already the doc_ids
        results = [(int(i), float(distances[0][j])) for j, i in enumerate(indices[0])]
        return results

    def search(self, query_embedding, k=10):
        ## Abstract method of the class
        distances, indices = self.index.search(query_embedding, k)
        return [(self.doc_ids[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]
    
    def score(self, query_text, top_k=10):
        if self.index is None:
            raise ValueError("FAISS index not generated")

        query_emb = self.model.encode([query_text], convert_to_tensor=True)
        query_np = query_emb.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_np)
        
        # Search FAISS index
        distances, indices = self.index.search(query_np, top_k)
        return [(self.doc_ids[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus_path= "/home/olagh48652/irg_course_assig/irg_final_project/dataset/LitSearch_corpus_clean"
    corpus = load_from_disk(corpus_path)

    dense_emb = DenseRetrieval(corpus["full"], device=device)

    faiss_path = "dataset/faiss_index_corpus_clean_full_dataset.index"
    dense_emb.get_embeddings(faiss_path=faiss_path)
                                
                            
if __name__ == "__main__":
    main()
