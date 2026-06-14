import chromadb, sys

def get_client():
    return chromadb.PersistentClient(path="chroma_db")

def get_collection(client, collection_name="papers"):
    return client.get_or_create_collection(name=collection_name)

def add_chunks(collection, chunks):
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(chunks)} chunks to ChromaDB")

def clear_collection(client, collection_name="papers"):
    client.delete_collection(name=collection_name)
    print(f"Cleared collection: {collection_name}")

if __name__ == "__main__":
    client = get_client()
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        clear_collection(client)
    else:
        collection = get_collection(client)
        print(f"Collection ready. Current count: {collection.count()}")