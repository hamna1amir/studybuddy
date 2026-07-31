# 📚 StudyBuddy — AI-Powered RAG Study Assistant

Upload a PDF, ask questions about it in plain language, and get answers grounded in the actual document — not a generic LLM guess. StudyBuddy turns static course material into an interactive study session using Retrieval-Augmented Generation (RAG).

[**🚀 Live Demo**](https://studybuddy-hamnaamir.streamlit.app/) · [Features](#-features) · [Tech Stack](#-tech-stack) · [How It Works](#-how-it-works) · [Getting Started](#-getting-started)

---

## 💡 Why StudyBuddy

Most "chat with your PDF" tools either hallucinate answers or choke on scanned documents. StudyBuddy is built to handle both problems directly:

- **Grounded answers** — every response is retrieved from the document's actual content via semantic search, not generated from the model's general knowledge alone.
- **OCR fallback** — scanned or image-based PDFs (lecture slides, photographed notes) are still readable, not just clean digital text.

## ✨ Features

- 📄 **PDF Upload & Ingestion** — drop in a PDF and StudyBuddy parses, chunks, and indexes it automatically
- 🔍 **OCR Support** — extracts text from scanned/image-based PDFs that standard text extraction can't read
- 🧠 **Semantic Search** — Sentence Transformer embeddings power similarity-based retrieval instead of simple keyword matching
- 🗂️ **Vector Database** — ChromaDB stores and indexes document embeddings for fast, persistent retrieval
- 💬 **Context-Aware Q&A** — ask questions in natural language and get answers grounded in the retrieved passages, powered by Groq's Llama 3
- 📝 **Summarization** — condense long documents or sections into digestible summaries

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / App** | Streamlit |
| **LLM Inference** | Groq (Llama 3) |
| **Embeddings** | Sentence Transformers |
| **Vector Database** | ChromaDB |
| **Document Ingestion** | OCR (Tesseract) + PDF text extraction |

## 🏗️ How It Works

```
studybuddy/
├── app.py                  # Streamlit entry point — UI and chat interface
├── ingest.py                # PDF loading and preprocessing
├── embed_and_store.py        # Chunking, embedding generation, ChromaDB storage
├── query.py                  # Semantic retrieval + Groq Llama 3 Q&A pipeline
├── summarize.py               # Document/section summarization
├── chroma_db/                  # Persistent vector store
├── data/                        # Uploaded/processed documents
└── .streamlit/                   # App configuration
```

**Pipeline:**
1. **Ingest** — PDF is loaded; OCR kicks in automatically for scanned pages
2. **Embed & Store** — text is chunked and converted to embeddings via Sentence Transformers, then stored in ChromaDB
3. **Query** — a user question is embedded and matched against stored chunks via semantic similarity search
4. **Answer** — the most relevant chunks are passed to Groq's Llama 3, which generates a grounded, context-aware response

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A free [Groq API key](https://console.groq.com/keys)
- Tesseract OCR installed locally (for scanned-PDF support)

### Installation

```bash
git clone https://github.com/hamna1amir/studybuddy.git
cd studybuddy
pip install -r requirements.txt
```

### Configure your API key

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### Run

```bash
streamlit run app.py
```

## ☁️ Deployment

Deployed on **Streamlit Community Cloud**:
🔗 **Live app:** [studybuddy-hamnaamir.streamlit.app](https://studybuddy-hamnaamir.streamlit.app/)

---

Built by **Hamna Amir** — a Retrieval-Augmented Generation study tool combining semantic search, OCR, and LLM-powered Q&A.
