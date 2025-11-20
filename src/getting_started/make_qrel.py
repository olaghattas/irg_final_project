from datasets import load_from_disk

qrel_path = "evaluation/litsearch.qrel"

dataset_query = load_from_disk("data/LitSearch_query")
with open(qrel_path, 'w', encoding='utf8') as outfile:
    for (ind,qline) in enumerate(dataset_query['full']):
        for id in qline['corpusids']:
            outfile.write(f"{ind} {0} {id} {qline['quality']}\n")