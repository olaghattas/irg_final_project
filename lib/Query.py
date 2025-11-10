from pydantic import BaseModel
from pydantic_core import from_json
from collections import Counter
from pyserini.index.lucene import LuceneIndexReader

class Query(BaseModel):
    contents: str

    def get_nnn(self, index_dir):
        index_reader = LuceneIndexReader(index_dir)
        analyzed = index_reader.analyze(self.contents)
        return dict(Counter(analyzed))
    
    # Define remaining TF-IDF embedding methods here
    

def main():
    query = Query.model_validate_json("{ \"contents\" : \"Robots are going to 3D imaging\" }")
    print(query.get_nnn("pyserini_index"))

    # You can add to this to see that you are getting what you expect

if __name__ == "__main__":
    main()