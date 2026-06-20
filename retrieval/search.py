from storage.vector_store import get_client, get_collection
from embeddings.embedder import embed_text

def search(query, k=5, source_filter=None):
    client = get_client()
    collection = get_collection(client)

    query_vector = embed_text(query)

    query_kwargs = {
        "query_embeddings" : [query_vector],
        "n_results" : k,
        "include" : ["documents","metadatas","distances"]
    }

    if source_filter:
        query_kwargs["where"] ={"source" : source_filter}

    results = collection.query(**query_kwargs)

    chunks = results["documents"][0]
    metadata = results["metadatas"][0]
    distance = results["distances"][0]

    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadata, distance)):
        score = 1-dist  # cosine distance to similarity
        print(f"\nResult {i+1} | Score: {score:.4f} | Source: {meta['source']}")
        print(chunk)

if __name__ == "__main__":
    search("Hardness of HEAs")