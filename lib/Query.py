from pydantic import BaseModel
from pydantic_core import from_json
from collections import Counter
from pyserini.index.lucene import LuceneIndexReader
from utils import get_dataset
import os

class Query(BaseModel):
    contents: str

    def get_nnn(self, index_dir):
        index_reader = LuceneIndexReader(index_dir)
        analyzed = index_reader.analyze(self.contents)
        return dict(Counter(analyzed))
    
    def get_bnn(self, index_dir):
        index_reader = LuceneIndexReader(index_dir)
        analyzed = index_reader.analyze(self.contents)
        return {term: 1 for term in set(analyzed)}
    
    # Define remaining TF-IDF embedding methods Here

  
def get_query():
    return get_dataset("query")


def main():
    query = Query.model_validate_json("{ \"contents\" : \"Robots are going to 3D imaging\" }")
    print(query.get_nnn("pyserini_index"))
    print(query.get_bnn("pyserini_index"))
    
    query2 = Query.model_validate_json("{ \"contents\" : \"Robots images are going robots to 3D imaging\" }")
    print(query2.get_nnn("pyserini_index"))
    print(query2.get_bnn("pyserini_index"))
    # You can add to this to see that you are getting what you expect

if __name__ == "__main__":
    main()