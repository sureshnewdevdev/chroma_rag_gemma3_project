# ChromaDB RAG + Gemma3:1b (Ollama) — Sample Project

This project demonstrates a complete **RAG (Retrieval-Augmented Generation)** pipeline where:

- **ChromaDB** is the **vector database (source of truth)** for retrieval
- **Ollama embeddings model** creates embeddings for documents and user queries
- **gemma3:1b** generates the final answer **only from retrieved context**

---

## 1) Prerequisites

### A) Install Python
- Recommended: **Python 3.10+**
- Verify:
  - Windows: `python --version`
  - Mac/Linux: `python3 --version`

### B) Install Ollama
1. Download and install Ollama (Windows/Mac/Linux)
2. Verify:
   - `ollama --version`

### C) Pull required Ollama models
Open terminal / cmd and run:

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

✅ `gemma3:1b` = generation (answer)  
✅ `nomic-embed-text` = embeddings (vector search)

---

## 2) Project Files

- `app.py` : Main application with line-by-line explanations (comments)
- `Templesofondipudur.txt` : Source data used for indexing
- `requirements.txt` : Python dependencies
- `chroma_db/` : Auto-created folder where Chroma stores vectors (created at runtime)

---

## 3) Setup (Step-by-step)

### Step 1: Extract the ZIP
Extract this zip to a folder, e.g.:
`C:\Projects\chroma_rag_gemma3_project`

### Step 2: Open terminal in that folder
Example:
```bash
cd C:\Projects\chroma_rag_gemma3_project
```

### Step 3: Create virtual environment (recommended)
**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
```

---

## 4) Run the project

```bash
python app.py
```

What happens when you run:
1. The file `Templesofondipudur.txt` is read
2. Text is split into chunks
3. Each chunk is embedded (vectorized) using `nomic-embed-text`
4. Chunks are stored in **ChromaDB** (persistent, saved in `chroma_db/`)
5. You can ask questions; the app retrieves top matches and uses **gemma3:1b** to answer.

---

## 5) Example Questions to Try

- `What is Mariamman temple worshipped for?`
- `Which temple has fire walking ritual?`
- `Which temple is related to Devanga community?`
- `What festival happens in Aadi Velli?`

Type `exit` to quit.

---

## 6) Notes / Troubleshooting

### If you get "model not found"
Run:
```bash
ollama list
```
If missing:
```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

### If you want to re-index from scratch
Delete the folder:
- `chroma_db/`
Then run again:
```bash
python app.py
```

### If port/service issues happen
Make sure Ollama is running (it runs as a background service after install).

---

## 7) Customization

In `app.py`, you can change:

- `SOURCE_FILE` to any text file
- `COLLECTION_NAME` to create a new collection
- `TOP_K` to retrieve more/less context
- `GEN_MODEL` to another Ollama chat model
- `EMBED_MODEL` to another Ollama embedding model
