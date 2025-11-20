import sys
sys.path.append('.')

from src.classes.query import Query
from src.classes.index import Index
from src.classes.document import Document
from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher


class TF_IDF(Index):

    def __init__(self, dir):
        self.dir = dir
        self.reader = LuceneIndexReader(dir)
        self.searcher = LuceneSearcher(dir)

    def search(self,query_embedding, doc_embedding_type ,k):
        # This is the naive implementation if it's too slow I can speed it up later
        if doc_embedding_type == 'nnn':
            docs = dict()
            for term in query_embedding.items():
                posting_list = self.reader.get_postings_list(term[0])
                for posting in posting_list:
                    if posting.docid in docs.keys():
                        docs[posting.docid] += term[1]*posting.tf
                    else:
                        docs[posting.docid] = term[1]*posting.tf
            out = sorted(docs.items(), key=lambda kv: kv[1], reverse=True)[0:k]
            # make sure to convert pyserini docids back into docids assigned in dataset
            return [(self.searcher.doc(item[0]).id(), item[1]) for item in out]
            # each of the remaining methods should return a list of (docid, score) pairs sorted by score
        else:
            return []
        
    def get_contents(self, d_ids):
        # I don't know why the searcher can't get the contents back on its own, but this is the best way I know how to do this for now
        return [Document.model_validate_json(self.searcher.doc(d_id).raw()).contents for d_id in d_ids]

def main():
    index = TF_IDF('indexes/pyserini_index')
    query = Query("Robots are going to 3D imaging")
    ranking = index.search(query.get_nnn(index.dir),'nnn', 10)
    print(ranking)
    texts = index.get_contents([rank[0] for rank in ranking])
    for (i,text) in enumerate(texts):
        print(f"---------------------------------RESULT {i}:--------------------------------------")
        print(text)
    return

if __name__ == "__main__":
    main()