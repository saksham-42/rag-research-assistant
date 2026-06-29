from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict

def split_documents(docs, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc.metadata["source"]].append(doc)

    chunks = []
    for source, pages in grouped.items():
        pages.sort(key=lambda d: d.metadata["page"])

        merged_text = "\n\n".join(page.page_content for page in pages)

        merged_doc = Document(
            page_content=merged_text,
            metadata={
                "source": source,
                "title": pages[0].metadata.get("title", ""),
                "first_page":str(pages[0].metadata.get("page", 0)),
                "last_page":str(pages[-1].metadata.get("page", 0)),
            }
        )

        chunks.extend(splitter.split_documents([merged_doc]))

    return chunks

if __name__ == "__main__":
    from ingestion.extractor import load_pdfs
    from ingestion.cleaner import clean_documents

    docs = load_pdfs()
    cleaned = clean_documents(docs)
    chunks = split_documents(cleaned)

    print(f"Cleaned pages: {len(cleaned)}")
    print(f"Chunks after splitting: {len(chunks)}")
    print(f"First chunk metadata: {chunks[0].metadata}")
    print(f"First chunk content:")
    print(chunks[0].page_content)