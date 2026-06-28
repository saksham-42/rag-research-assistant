from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from retrieval.search import retriever
from storage.vector_store import get_vector_store

def hybrid_retriever(k=5, source=None):
    vector_store = get_vector_store()
    all_docs = vector_store.get()
    docs = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    
    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(docs, metadatas)
    ]
    
    if source:
        documents = [d for d in documents if d.metadata.get("source") == source]
    
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)
    vector_retriever = retriever(k=k, source=source)
    
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

if __name__ == "__main__":
    hybrid = hybrid_retriever()
    results = hybrid.invoke("What manufacturing methods are used for High Entropy Alloys?")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1} | Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(f"{doc.page_content}")