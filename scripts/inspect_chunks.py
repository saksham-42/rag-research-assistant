import sys
from storage.vector_store import get_client, get_collection

def inspect_chunks(every_nth=10, source=None):
    client = get_client()
    collection = get_collection(client)

    total = collection.count()
    print(f"Total chunks in ChromaDB: {total}\n")

    results = collection.get(
        where={"source": source} if source else None,
        include=["documents", "metadatas"]
    )

    chunks = list(zip(results["ids"], results["documents"], results["metadatas"]))

    for i, (chunk_id, text, metadata) in enumerate(chunks):
        if i % every_nth == 0:
            print(f"--- Chunk {i} ---")
            print(f"ID: {chunk_id}")
            print(f"Metadata: {metadata}")
            print(f"Text preview: {text[:300]}")
            print()

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else None
    inspect_chunks(every_nth=10, source=source)