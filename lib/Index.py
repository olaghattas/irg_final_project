from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher

class Index:
    def __init__(self, dir):
        self.dir = dir

    def search(self, query):
        # Return a vector of embeddings for best documents
        pass