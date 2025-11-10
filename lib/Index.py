from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher
from abc import ABC, abstractmethod

class Index(ABC):
    def __init__(self, dir):
        self.dir = dir

    @abstractmethod
    def search(self, query_embedding):
        
        # Return a vector of embeddings for best documents
        pass