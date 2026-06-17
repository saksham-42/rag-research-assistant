# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) system for querying materials science research papers. Built as a portfolio project over 6 weeks, covering the full pipeline from PDF ingestion to semantic search, LLM-powered answers, and cloud deployment.

---

## Project Status

| Week | Focus | Status |
|---|---|---|
| 1 | PDF ingestion, cleaning, chunking, vector storage | ✅ Complete |
| 2 | Embeddings, semantic search, query pipeline | ⬜ Upcoming |
| 3 | LlamaIndex integration, query engine | ⬜ Upcoming |
| 4 | FastAPI backend, evaluation metrics | ⬜ Upcoming |
| 5 | Streamlit UI, Docker, hallucination detection | ⬜ Upcoming |
| 6 | Deployment, CI/CD, resume prep | ⬜ Upcoming |

---

## Tech Stack (Week 1)

| Component | Tool |
|---|---|
| PDF Extraction | PyMuPDF |
| Text Cleaning | Python (re, nltk) |
| Tokenization | tiktoken (cl100k_base) |
| Vector Database | ChromaDB |

---

## Project Structure

```
rag-research-assistant/
├── data/                    # PDF papers (not tracked in git)
├── output/                  # Extracted and cleaned text files (not tracked in git)
├── chroma_db/               # Vector database (not tracked in git)
├── ingestion/
│   ├── extractor.py         # PDF → raw text
│   ├── cleaner.py           # raw text → clean text
│   ├── chunker.py           # fixed-size chunking (512 tokens, 50 overlap)
│   └── sen_chunking.py      # sentence-aware chunking (final strategy)
├── storage/
│   └── vector_store.py      # ChromaDB setup, add chunks, clear collection
├── scripts/
│   └── inspect_chunks.py    # inspect stored chunks and metadata
├── ingest.py                # full pipeline — runs all PDFs through ingestion
├── .env                     # API keys (not tracked in git)
├── .gitignore
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/saksham-42/rag-research-assistant.git
cd rag-research-assistant
```

**2. Create virtual environment**
```bash
python -m venv env
env\Scripts\activate        # Windows
source env/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install pymupdf chromadb tiktoken nltk python-dotenv
```

**4. Download NLTK data**
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

**5. Add your PDFs**
```bash
mkdir data output
```
Put your PDF files in the `data/` folder.

**6. Run ingestion**
```bash
python ingest.py data
```

**7. Inspect chunks**
```bash
python -m scripts.inspect_chunks
```

---

## Ingestion Pipeline

```
PDF
 └─→ extractor.py       →  raw text + [Page X] markers saved to output/
      └─→ cleaner.py    →  removes headers, footers, unicode noise, citations
           └─→ sen_chunking.py   →  sentence-aware chunks with metadata
                └─→ vector_store.py   →  stored in ChromaDB
```

Each chunk carries metadata:
- `source` — filename of the original PDF
- `page` — page number the chunk came from
- `chunk_index` — position of the chunk within the document

---

## Chunking Strategy

Two strategies were implemented and compared:

**Fixed-size chunking** — splits at exactly 512 tokens with 50-token overlap. Fast and predictable but cuts mid-sentence.

**Sentence-aware chunking** — splits at sentence boundaries, never cutting mid-sentence. Slightly fewer chunks, better quality for retrieval.

**Decision: sentence-aware chunking is used in the final pipeline.**

Reason: materials science papers contain precise numerical values and citations tightly coupled to their sentences. Cutting mid-sentence loses context that directly affects retrieval quality. For example, fixed-size chunking split *"the UTS of which can reach"* from *"1100 MPa [32]"* into two separate chunks — neither alone answers the question.

**Known limitation:** tables are not extracted as structured data. PyMuPDF flattens table content into raw text, losing column relationships. Noted for future improvement.

---

## Week 1 — What was built

- PDF extraction handling clean, multi-column, and scanned PDFs
- Text cleaning pipeline removing headers, footers, unicode artifacts, citation blocks
- Fixed-size and sentence-aware chunking with page number metadata
- ChromaDB vector store with duplicate detection
- Full batch ingestion pipeline for entire paper folders
- 726 chunks from 22 materials science papers (1 scanned PDF skipped)

**Papers cover:** titanium alloys, high entropy alloys (HEAs), dual-phase steels, martensitic steels, aluminum alloys

---

## Re-ingestion

If you change chunking strategy or cleaning logic, wipe ChromaDB and re-ingest:

```bash
python storage/vector_store.py --clear
python ingest.py data
```