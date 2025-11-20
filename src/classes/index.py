from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher
from abc import ABC, abstractmethod
import os
import subprocess

class Index(ABC):
    def __init__(self, dir):
        self.dir = dir

    @abstractmethod
    def search(self, query_embedding, doc_embedding_type, k):
        
        # Return a list of the top k (document id,rank score) pairs
        pass
    
    
    ## will need to be put in a different function
    @staticmethod
    def build_pyserini_index(input_dir, index_dir, threads=4):
        """
        Build a Pyserini Lucene index from JSON documents.
        """
        
        os.makedirs(os.path.dirname(index_dir), exist_ok=True)

        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", input_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads),
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]

        print(f"Building Pyserini index from {input_dir} → {index_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(" Indexing failed:")
            print(result.stderr)
            raise RuntimeError("Pyserini indexing failed")
        else:
            print(" Index successfully created!")
            print(result.stdout)
            
            