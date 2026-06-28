from storage.vector_store import get_vector_store

def retriever(k=5, source=None):
    vector_store = get_vector_store()
    search_k = {"k":k}
    if source:
        search_k["filter"] = {"source": source}
    return vector_store.as_retriever(search_kwargs = search_k)

def search(query, k=5, source_filter=None):
    retrieve = retriever(k=k, source = source_filter)
    results = retrieve.invoke(query)

    for i, doc in enumerate(results):
        print(f"\n Results {i+1} | Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(doc.page_content[:500])
    return results

def search_scores(query, k=5, source_filter=None):
    vector_store = get_vector_store()
    filter_k = {"source": source_filter} if source_filter else None
    results = vector_store.similarity_search_with_score(query, k=k, filter=filter_k)
    for i, (doc, dist) in enumerate(results):
        score = 1-dist
        print(f"\n Results {i+1} | Score: {score:.4f} |Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(doc.page_content)
    return results

if __name__ == "__main__":
    search_scores("What manufacturing methods are used for High Entropy Alloys?")