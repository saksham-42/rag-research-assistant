import os, shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from generation.generator import ask
from ingestion.cleaner import clean_documents
from ingestion.extractor import load_pdfs
from ingestion.sen_chunking import split_documents
from storage.vector_store import get_vector_store, build_vector_store
from app.schemas import QueryRequest, QueryResponse, IngestResponse, DocumentResponse, DeleteResponse
from app.logging import logger

app = FastAPI(title="RAG RESEARCH ASSISTANT")

cache = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)

logger.info("We are online now!")
@app.get("/")
def root():
    return {"message": "RAG Research Assistant API", "docs" : "/docs"}

@app.post("/upload_file", response_model=IngestResponse)
async def ingest(file:UploadFile = File()):
    logger.info(f"Ingest request received:{file.filename}")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    os.makedirs("data", exist_ok=True)
    
    vs = get_vector_store()
    exists = vs.get()
    for m in exists["metadatas"]:
        if file.filename in m.get("source",""):
            raise HTTPException(status_code=409, detail=f"{file.filename} is already ingested")

    save_path = os.path.join("data",file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        docs = load_pdfs()
        clean = clean_documents(docs)
        chunks = split_documents(clean)

        new_chunks = [c for c in chunks if file.filename in c.metadata.get("source", "")]
        if not new_chunks:
            raise HTTPException(status_code=500, detail="No chunks were extracted from PDF")
        build_vector_store(new_chunks)
        logger.info(f"Ingested {len(new_chunks)} chunks from {file.filename}")
        return IngestResponse(filename= file.filename, chunks_ingested = len(new_chunks))
    except Exception as e:
        logger.error(f"Error in Ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest):
    logger.info(f"Query Received: {payload.question}")
    ques = payload.question.strip()
    if not ques:
        raise HTTPException(status_code=400, detail="Question can't be empty")
    ques = f"{ques}::{payload.filter_document or 'all'}"
    if ques in cache:
        logger.info("Cache hit — returning cached answer")
        return QueryResponse(**cache[ques])
    try:
        ans = ask(ques,k=5,source=payload.filter_document)
        cache[ques] = ans
        return QueryResponse(**ans)
    except Exception as e:
        logger.error(f"Error in Query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/documents", response_model=DocumentResponse)
def all_documents():
    logger.info("Document list requested")
    try:
        vs = get_vector_store()
        all_docs = vs.get()
        seen = set()
        docs_list = []
        for meta in all_docs["metadatas"]:
            src = meta.get("source", "")
            if src not in seen:
                seen.add(src)
                docs_list.append({
                    "source": src,
                    "title": meta.get("title", os.path.basename(src))
                })
        return DocumentResponse(documents=docs_list)
    except Exception as e:
        logger.error(f"Error in Documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{name}", response_model=DeleteResponse)
def delete_document(name: str):
    logger.info(f"Delete request for: {name}")
    try:
        vs = get_vector_store()
        all_docs = vs.get()
        ids_to_delete = [
            id_ for id_, meta in zip(all_docs["ids"], all_docs["metadatas"])
            if name in meta.get("source", "")
        ]
        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"No document found matching '{name}'.")
        vs.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks from {name}")
        return DeleteResponse(deleted=name, chunks_removed=len(ids_to_delete))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Deleting: {e}")
        raise HTTPException(status_code=500, detail=str(e))