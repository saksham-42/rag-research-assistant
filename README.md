# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) system for querying materials science research papers. Built as a portfolio project, covering the full pipeline from PDF ingestion to semantic search, LLM-powered answers, hybrid retrieval, and guaranteed results via web fallback.

---

## What It Does

- Upload materials science research papers (PDFs)
- Ask technical questions in natural language
- Get detailed, cited answers grounded in the papers — with paper title and page range
- If the answer isn't in the corpus, automatically falls back to Semantic Scholar / ArXiv and returns relevant paper links
- Never returns "I don't know" — always produces a result

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF Extraction | PyMuPDF |
| Text Cleaning | Python (re, nltk) |
| Chunking | Sentence-aware (RecursiveCharacterTextSplitter) |
| Embeddings | Google Gemini (`gemini-embedding-2`) |
| Vector Database | ChromaDB (cosine similarity) |
| Keyword Search | BM25 (via langchain-community) |
| Hybrid Retrieval | EnsembleRetriever (BM25 + vector, 40/60) |
| LLM | Google Gemini (`gemini-2.5-flash-lite`) via LangChain |
| Web Fallback | Semantic Scholar API + ArXiv API |
| Framework | LangChain |

---

## Project Structure

```
rag-research-assistant/
├── app/
│   ├── logging.py               # Logger setup
│   ├── main.py                  # FastAPI app, endpoints, caching
│   └── schemas.py               # Pydantic request/response models
├── data/                        # PDF papers (not tracked in git)
├── chroma_db/                   # Vector database (not tracked in git)
├── embeddings/
│   └── embedder.py              # Gemini embedding model
├── evaluation/
│   ├── ans_eval.py              # answer quality scoring (LLM-as-judge)
│   └── eval.py                  # retrieval precision measurement
├── generation/
│   └── generator.py             # LLM chain
├── ingestion/
│   ├── cleaner.py               # removes headers, footers, noise
│   ├── extractor.py             # PDF to raw text pages
│   └── sen_chunking.py          # sentence-aware chunking with metadata
├── retrieval/
│   ├── fallback.py              # Semantic Scholar + ArXiv fallback
│   ├── hybrid.py                # BM25 + vector ensemble retriever
│   └── search.py                # vector search with optional source filter
├── scripts/
│   └── inspect_chunks.py        # chunk inspection utility
├── storage/
│   └── vector_store.py          # ChromaDB setup and retrieval
├── ingest.py                    # full ingestion pipeline
├── test.py                      # 83-question end-to-end test
└── ui.py                        # Streamlit frontend
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
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
SEMANTIC_API_KEY=your_key_here   # optional, increases rate limits
```

**5. Add your PDFs**

Put PDF files in the `data/` folder.

**6. Run ingestion**
```bash
python -m ingest
```

**7. Ask a question**
```bash
python -m generation.generator
```

---

## Retrieval Strategy

**Hybrid retrieval** — combines BM25 keyword search with vector semantic search using `EnsembleRetriever` (50/50 weights). BM25 handles exact term matches; vector search handles semantic similarity.

**Two-stage result guarantee:**
1. Embed the query, check top similarity score against threshold (0.7)
2. If score above threshold → answer from corpus with citations
3. If score below threshold → fall back to Semantic Scholar, then ArXiv

---

## Citation System

Every answer includes inline citations and a structured sources section:

```
Sources:
  [1] Effect of solution treatment and aging... — Pages [0-3] — data\Al-Mg-Si-AM.pdf
```

---

## Chunking Strategy

Sentence-aware chunking — whole papers are merged per source, then split at sentence boundaries. Never cuts mid-sentence.

**Known limitation:** tables are not extracted as structured data. PyMuPDF flattens table content into raw text, losing column relationships.

---

## Corpus

20 materials science papers covering:
- Titanium alloys (Ti-6Al-4V, TiAl, Ti-V omega phase)
- High entropy alloys (HEAs) — CoCrNi, FeCoNi, CoNiAl systems
- Aluminum alloys (Al-Mg-Si, Al-Zn-Mg-Cu)
- Steels (dual-phase, martensitic, 12% Cr)

546 chunks after ingestion and noise filtering.

---

## Evaluation Results

### Retrieval Precision (chunk_size=2000, 83 questions)
| Metric | Score |
|---|---|
| Hit Rate @3 | 63.9% |
| Mean Precision @3 | 40.2% |

### Answer Quality (30 questions, LLM-as-judge)
| Metric | Score |
|---|---|
| Avg Faithfulness | 3.23 / 5 |
| Avg Relevance | 4.93 / 5 |

### Chunk Size Ablation Study
| Metric | chunk_size=1000 | chunk_size=2000 |
|---|---|---|
| Hit Rate @3 | 66.3% | 63.9% |
| Mean Precision @3 | 37.3% | 40.2% |
| Avg Faithfulness | 3.20 / 5 | 3.23 / 5 |
| Avg Relevance | 4.73 / 5 | 4.93 / 5 |

**Decision:** chunk_size=2000 selected as default. While smaller chunks achieved a marginally higher hit rate, chunk_size=2000 produced better precision, faithfulness, and relevance — critical for a research assistant where answer completeness matters more than raw retrieval coverage.

### Retrieval-Weight Tuning
**Fix applied:** Adjusted hybrid retrieval weights from 0.5/0.5 to 0.4/0.6 (BM25/vector), giving more weight to semantic search.

| Metric | Before (0.5/0.5) | After (0.4/0.6) | Change |
|---|---|---|---|
| Hit Rate @3 | 63.9% | 77.1% | +13.2% |
| Mean Precision @3 | 40.2% | 49.4% | +9.2% |
| Avg Faithfulness | 3.23 / 5 | 3.50 / 5 | +0.27 |
| Avg Relevance | 4.93 / 5 | 4.87 / 5 | -0.06 |

**Remaining failures:** A small set of HEA-specific questions (CoCrNi, CoCrNiAlCu papers) still miss due to semantic similarity between HEA papers making them hard to distinguish at retrieval time.


---

## Test Results

83-question end-to-end test across all alloy systems:
- **56/83** answered from corpus with citations
- **27/83** correctly routed to web fallback which including a few partial corpus answers with citations

---

## Re-ingestion

```bash
Remove-Item -Recurse -Force chroma_db
python -m ingest
```

**Note:** Don't switch embedding models without full re-ingestion. Mixed embeddings from different models produce incompatible vector spaces and break similarity scores.