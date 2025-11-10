from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import faiss
import os

class Embedding(ABC):
    def __init__(self, rep):
        self.rep = rep
    
    @abstractmethod
    def score(query):
        pass

    
class DenseEmbedding(Embedding):
    def __init__(self, corpus_dataset, model_name="all-MiniLM-L6-v2", device=None):
        # Load the pre-trained model
        self.model = SentenceTransformer(model_name, device=device)
        self.corpus_dataset = corpus_dataset
        
        self.rep = None
        self.index = None
        self.dim = None

        self.corpus_dataset = corpus_dataset 
        self.doc_ids = [doc["corpusid"] for doc in corpus_dataset]
        
    def generate_embeddings(self, faiss_path = None):
        
        # Combine title + abstract (and optionally full paper) per document
        corpus_texts = [
            doc["title"] + ". " + doc["abstract"] for doc in self.corpus_dataset
        ]
        
        self.rep = self.model.encode(corpus_texts, batch_size=64, convert_to_tensor=True, show_progress_bar=True)

        ## using FAISS for saving
        corpus_np = self.rep.cpu().numpy().astype('float32')
        
        # Dimension of  embedding
        self.dim = corpus_np.shape[1]
        
        ## available index
        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        ## Exact Search for Inner Product
        self.index = faiss.IndexFlatIP(self.dim)  
        ## varant Exact Search for L2
        # self.index = faiss.IndexFlatL2(self.dim)  
        
        faiss.normalize_L2(corpus_np) # normalize for cosine similarity
        self.index.add(corpus_np)
        
        print(f"Total sentences indexed: {index.ntotal}")

        # Save FAISS index if path provided
        if faiss_path:
            faiss.write_index(self.index, faiss_path)
            print(f"Saved FAISS index to {faiss_path}")
            
    def get_embeddings(self, faiss_path=None):
        if faiss_path and os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
            print(f"Loaded FAISS index from {faiss_path}")
        else:
            print("Generating embeddings and FAISS index...")
            self.generate_embeddings(faiss_path=faiss_path)
            
    # ## temp
    # def lookup_():
    #     corpus_dataset.set_format("pandas")

    #     df = corpus_dataset.to_pandas().set_index("corpusid")
    #     df.loc[7819967]
        
    def score(self, query_text, top_k=10):
        if self.index is None:
            raise ValueError("FAISS index not generated")

        query_emb = self.model.encode([query_text], convert_to_tensor=True)
        query_np = query_emb.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_np)
        
        # Search FAISS index
        distances, indices = self.index.search(query_np, top_k)
        return [(self.doc_ids[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]