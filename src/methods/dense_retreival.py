from datasets import load_from_disk

corpus_dataset = load_from_disk("dataset/LitSearch_corpus_clean")["full"]

dense_emb = DenseEmbedding(corpus_dataset)
dense_emb.get_embeddings(faiss_path="dataset/LitSearch_corpus_clean.faiss")

query = "Robots for 3D imaging"
top_docs = dense_emb.score(query, top_k=5)

for docid, score in top_docs:
    print(f"{docid}: {score:.4f}")