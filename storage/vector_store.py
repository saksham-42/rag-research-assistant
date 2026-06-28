from langchain_chroma import Chroma
from embeddings.embedder import embedding_model

def build_vector_store(chunks):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="papers",
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "cosine"}
    )

def get_vector_store():
    return Chroma(
        collection_name="papers",
        embedding_function=embedding_model,
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "cosine"}
    )

if __name__ == "__main__":
    from ingestion.extractor import load_pdfs
    from ingestion.cleaner import clean_documents
    from ingestion.sen_chunking import split_documents

    docs = load_pdfs()
    cleaned = clean_documents(docs)
    chunks = split_documents(cleaned)

    test_chunks = [c for c in chunks if "Al-Mg-Si-AM" in c.metadata["source"]]
    print(f"Testing with {len(test_chunks)} chunks from one paper")

    vs = build_vector_store(test_chunks)
    print(f"Stored. Collection count: {vs._collection.count()}")