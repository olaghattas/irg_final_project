import sys
sys.path.append('.')

from src.classes.colbert import ColBert
from datasets import load_from_disk
from src.classes.query import Query

COLBERT_INDEX_LOCATION = "indexes/colbert_index"
QUERY_LOCATION = "data/LitSearch_query"
RUN_LOCATION = "run_files/colbert.run"

def main():
    index = ColBert(COLBERT_INDEX_LOCATION)
    dataset = load_from_disk(QUERY_LOCATION)
    with open(RUN_LOCATION, 'w', encoding='utf8') as run_file:
        for (i,qline) in enumerate(dataset['full']):
            query = Query(qline['query'])
            ranking = index.search(query.get_colbert(index.model),'',50)
            for (j,rank) in enumerate(ranking):
                run_file.write(f"{i} Q0 {rank[0]} {j+1} {rank[1]} colbert\n")
    return

if __name__ == '__main__':
    main()