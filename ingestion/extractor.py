from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

def load_pdfs(folder_path="data"):
    loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyMuPDFLoader, loader_kwargs={"flags": 4})
    return loader.load()

if __name__ == "__main__":
    docs = load_pdfs()
    print(f"Loaded {len(docs)} page-level Document objects")
    print(f"First doc metadata: {docs[0].metadata}")