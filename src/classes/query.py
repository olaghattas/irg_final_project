import sys
sys.path.append('.')

from collections import Counter
from pyserini.index.lucene import LuceneIndexReader
from src.utils import get_dataset
import os
import numpy as np
import nltk

class Query:
    contents: str

    def __init__(self,contents):
        self.contents = contents

    def get_nnn(self, index_dir):
        index_reader = LuceneIndexReader(index_dir)
        analyzed = index_reader.analyze(self.contents)
        return dict(Counter(analyzed))
    
    def get_bnn(self, index_dir):
        index_reader = LuceneIndexReader(index_dir)
        analyzed = index_reader.analyze(self.contents)
        return {term: 1 for term in set(analyzed)}
    
    def get_colbert(self, model):
        return np.array(model.encode(nltk.sent_tokenize(self.contents)))
    
    # Define remaining TF-IDF embedding methods Here

  
def get_query():
    return get_dataset("query")


def main():
    query = Query("Robots are going to 3D imaging")
    print(query.get_nnn("indexes/pyserini_index"))
    print(query.get_bnn("indexes/pyserini_index"))
    
    query2 = Query("Robots images are going robots to 3D imaging")
    print(query2.get_nnn("indexes/pyserini_index"))
    print(query2.get_bnn("indexes/pyserini_index"))
    # You can add to this to see that you are getting what you expect

if __name__ == "__main__":
    main()