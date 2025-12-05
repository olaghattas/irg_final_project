with open("run_files/bm25_krovetz_llm_rerank.run", 'r', encoding= 'utf8') as infile:
    with open("run_files/bm25_krovetz_llm_rerank2.run", 'w', encoding= 'utf8') as outfile:
        lines = infile.readlines()
        query = "0"
        docids = []
        for line in lines:
            items = line.split()
            if items[0] == query:
                if items[2] not in docids:
                    docids.append(items[2])
                    outfile.write(line)
                else:
                    pass
            else:
                docids.clear()
                query = items[0]
