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

    if not documents:
        documents = [
            Document(page_content=doc, metadata= meta)
            for doc, meta in zip(docs, metadatas)
        ]
    
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)
    vector_retriever = retriever(k=k, source=source)
    
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

def retrieve(query: str, k: int = 5, source=None) -> dict:
    hybrid = hybrid_retriever(k=k, source=source)
    docs = hybrid.invoke(query)

    vector_store = get_vector_store()
    scored = vector_store.similarity_search_with_score(query, k=1)
    best_score = (1 - scored[0][1]) if scored else 0.0

    return {
        "documents": docs,
        "score": best_score
    }

if __name__ == "__main__":
    hybrid = retrieve("What manufacturing methods are used for High Entropy Alloys?")
    for i, doc in enumerate(hybrid["documents"]):
        print(f"\nResult {i+1} | Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(f"{doc.page_content}")