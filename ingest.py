import time
from collections import defaultdict
from ingestion.extractor import load_pdfs
from ingestion.cleaner import clean_documents
from ingestion.sen_chunking import split_documents
from storage.vector_store import get_vector_store

BATCH_SIZE = 35
SLEEP_TIME = 35

def is_noise_chunk(text):
    return len(text.split()) < 15

def already_ingested(vector_store, source):
    results = vector_store._collection.get(where={"source": source}, limit=1)
    return len(results["ids"]) > 0

def ingest(folder_path="data"):
    print("Loading PDFs...")
    docs = load_pdfs(folder_path)
    print(f"Loaded {len(docs)} pages")

    print("Cleaning...")
    cleaned = clean_documents(docs)
    print(f"{len(cleaned)} pages after cleaning")

    print("Splitting into chunks...")
    chunks = split_documents(cleaned)
    print(f"{len(chunks)} chunks before noise filter")

    chunks = [c for c in chunks if not is_noise_chunk(c.page_content)]
    print(f"{len(chunks)} chunks after noise filter")

    grouped = defaultdict(list)
    for c in chunks:
        grouped[c.metadata["source"]].append(c)

    vector_store = get_vector_store()

    for source, paper_chunks in grouped.items():
        if already_ingested(vector_store, source):
            print(f"Skipping {source} - already in ChromaDB")
            continue

        print(f"Ingesting {source} ({len(paper_chunks)} chunks)...")
        total_batches = -(-len(paper_chunks) // BATCH_SIZE)

        for i in range(0, len(paper_chunks), BATCH_SIZE):
            batch = paper_chunks[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            print(f"  Batch {i // BATCH_SIZE + 1}/{total_batches} stored")
            time.sleep(SLEEP_TIME)

        print(f"Done with {source}")

    print(f"Total chunks stored: {vector_store._collection.count()}")

if __name__ == "__main__":
    ingest()