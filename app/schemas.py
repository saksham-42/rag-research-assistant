from pydantic import BaseModel, Field
from typing import Optional

class IngestResponse(BaseModel):
    filename: str
    chunks_ingested:int

class Citations(BaseModel):
    title: str
    source: str
    first_page: int|str
    last_page: int|str

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_history: list = []
    filter_document: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citations]
    chunks_used: int

class DocumentItem(BaseModel):
    source: str
    title: str

class DocumentResponse(BaseModel):
    documents: list[DocumentItem]

class DeleteResponse(BaseModel):
    deleted:str
    chunks_removed: int