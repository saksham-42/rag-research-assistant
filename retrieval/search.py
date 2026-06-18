from storage.vector_store import get_client, get_collection
from embeddings.embedder import embed_text

def search(query, k=10):
    client = get_client()
    collection = get_collection(client)

    query_vector = embed_text(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results = k,
        include = ["documents","metadatas","distances"] 
    )

    chunks = results["documents"][0]
    metadata = results["metadatas"][0]
    distance = results["distances"][0]

    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadata, distance)):
        score = 1-dist  # cosine distance to similarity
        print(f"\nResult {i+1} | Score: {score:.4f} | Source: {meta['source']} | Page : {meta['page']}")
        print(chunk)

if __name__ == "__main__":
    search("Hardness of HEAs")