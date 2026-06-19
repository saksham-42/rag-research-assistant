import os, sys, time, re
from ingestion.extractor import extract_pdf
from ingestion.cleaner import clean_text
from ingestion.sen_chunking import sent_chunker
from storage.vector_store import add_chunks, get_client, get_collection
from embeddings.embedder import embed_texts

batch_size = 35

def is_noise_chunk(text):
    words = len(text.split())
    return words<30

def already_ingested(collection, filename):
    results = collection.get(where = {"source": filename})
    return len(results["ids"])>0

def ingest_folder(folder_path):
    client = get_client()
    collection = get_collection(client)

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        return print("No pdf files found")
    
    for curr_pdf in pdf_files:
        pdf_path = os.path.join(folder_path, curr_pdf)
        clean_file = curr_pdf.replace(".pdf", "-clean.txt")

        if already_ingested(collection, clean_file):
            print(f"Skipping {curr_pdf} - Already registered")
            continue

        print (f"Ingesting {curr_pdf}..")

        raw_text = extract_pdf(pdf_path)

        if not raw_text.strip():
            print(f"Skipping {curr_pdf} - No text extracted")
            continue

        cleaned = clean_text(raw_text)
        chunks = sent_chunker(cleaned, clean_file)

        chunks = [c for c in chunks if not is_noise_chunk(c["text"])]

        all_embeddings = []
        total_batch = -(-len(chunks)//batch_size)
        for i in range(0,len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c["text"] for c in batch]
            embeddings = embed_texts(texts)
            all_embeddings.extend(embeddings)
            print(f"Embedded batch {i//batch_size+1}/{total_batch}")
            time.sleep(25)

        add_chunks(collection, chunks, all_embeddings)
        print(f"Done, {len(chunks)} chunks stored")

    print(f"Total in ChromaDB: {collection.count()}")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv)> 1 else "data"
    ingest_folder(folder) 

    
