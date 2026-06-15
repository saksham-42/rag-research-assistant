import os, sys
from ingestion.extractor import extract_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunking
from storage.vector_store import add_chunks, get_client, get_collection

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

        print (f"Ingested {curr_pdf}..")

        raw_text = extract_pdf(pdf_path)

        if not raw_text.strip():
            print(f"Skipping {curr_pdf} - No text extracted")
            continue

        cleaned = clean_text(raw_text)
        chunks = chunking(cleaned, clean_file)
        add_chunks(collection, chunks)
        
        print(f"Done, {len(chunks)} chunks stored")

    print(f"Total in ChromaDB: {collection.count()}")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv)> 1 else "data"
    ingest_folder(folder) 

    
