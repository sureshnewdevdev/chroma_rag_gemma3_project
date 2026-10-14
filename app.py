# ==========================================================
# app.py  (ChromaDB as source + RAG + Gemma3:1b via Ollama)
# ==========================================================
#
# Goal:
# 1) Read a source text file (Templesofondipudur.txt)
# 2) Split it into chunks
# 3) Create embeddings for each chunk using Ollama embeddings
# 4) Store (upsert) those chunks into ChromaDB (persistent)
# 5) For a user question:
#    - embed the question
#    - retrieve top-matching chunks from ChromaDB
#    - send retrieved context + question to gemma3:1b (Ollama)
#    - print the answer
#
# NOTE:
# - ChromaDB stores vectors + metadata; it becomes your "knowledge source".
# - gemma3:1b is used ONLY for generation (answer writing).
# - An embeddings model is used ONLY for vector creation (search).
#
# ----------------------------------------------------------

# Import os for file paths and folder checks
import os

# Import uuid to create unique IDs if needed
import uuid

# Import dataclass to define a simple "Document" structure
from dataclasses import dataclass

# Import List for readable type hints
from typing import List, Tuple

# Import chromadb client to store and search vectors
import chromadb

# Import Ollama Python SDK to call local models (chat + embeddings)
import ollama


# -------------------------------
# CONFIGURATION (EDIT IF NEEDED)
# -------------------------------

# Chroma will store its database files inside this folder (created automatically)
CHROMA_DIR = "chroma_db"

# Chroma collection name (like a table name in a DB)
COLLECTION_NAME = "ondipudur_temples"

# Input knowledge file (your source content)
SOURCE_FILE = "Templesofondipudur.txt"

# Embeddings model (Ollama) to turn text into vectors
# If you don't have it, the README shows how to pull it.
EMBED_MODEL = "nomic-embed-text"

# LLM model (Ollama) to generate final answers
GEN_MODEL = "gemma3:1b"

# Chunking rules: how to split a long file into smaller pieces
MIN_CHUNK_LEN = 40  # ignore tiny chunks (noise)
TOP_K = 3           # how many chunks to retrieve for each question


# -------------------------------
# DATA STRUCTURE
# -------------------------------

# A document chunk stored in Chroma
@dataclass
class Document:
    id: str          # unique id
    text: str        # chunk content
    source: str = "" # metadata: where it came from


# -------------------------------
# STEP 1: LOAD + CHUNK TEXT FILE
# -------------------------------

def load_and_chunk_text(filepath: str) -> List[Document]:
    """
    Reads a text file and splits it into chunks.
    Returns a list of Document chunks.
    """

    # Open the file using UTF-8 (safe for Indian language/Unicode text too)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by blank lines (two new lines) to get paragraph-like chunks
    raw_chunks = [c.strip() for c in content.split("\n\n")]

    # Keep only chunks that are meaningful (avoid very short lines)
    chunks = [c for c in raw_chunks if len(c) >= MIN_CHUNK_LEN]

    # Prepare list of Document objects
    docs: List[Document] = []

    # Convert each chunk into a Document with a stable ID
    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                id=f"chunk_{i}",            # stable id so upsert won't duplicate
                text=chunk,                 # the chunk text
                source=os.path.basename(filepath)  # filename as source
            )
        )

    # Return list of documents
    return docs


# -------------------------------
# STEP 2: GET EMBEDDING FROM OLLAMA
# -------------------------------

def get_embedding(text: str) -> List[float]:
    """
    Calls Ollama embeddings endpoint and returns a vector for the given text.
    This vector is what Chroma stores and compares for similarity search.
    """

    # Ask Ollama to generate embeddings using the chosen embedding model
    # response["embedding"] is a list[float]
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)

    # Return the embedding vector
    return response["embedding"]


# -------------------------------
# STEP 3: SET UP CHROMA (PERSISTENT)
# -------------------------------

def get_chroma_collection():
    """
    Creates/loads a persistent ChromaDB collection.
    'PersistentClient' stores DB files on disk so data stays after program exit.
    """

    # Create (or reuse) a persistent Chroma client
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Create or load the collection by name
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Return collection object
    return collection


# -------------------------------
# STEP 4: ADD/UPSERT DOCUMENTS INTO CHROMA
# -------------------------------

def upsert_documents(collection, docs: List[Document]) -> None:
    """
    Upserts documents into ChromaDB:
    - ids: unique ids
    - documents: raw text
    - embeddings: vector representations
    - metadatas: extra info like source
    """

    # Build arrays required by Chroma
    ids = []         # list of IDs
    texts = []       # list of document texts
    embeds = []      # list of embedding vectors
    metas = []       # list of metadata dicts

    # Loop through documents and prepare data for Chroma
    for d in docs:
        ids.append(d.id)                     # ID for each chunk
        texts.append(d.text)                 # raw chunk
        embeds.append(get_embedding(d.text)) # embedding vector for chunk
        metas.append({"source": d.source})   # metadata

    # Upsert means: insert new OR update existing by ID (no duplicates)
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeds,
        metadatas=metas
    )


# -------------------------------
# STEP 5: RETRIEVE TOP-K CONTEXT FROM CHROMA
# -------------------------------

def retrieve_context(collection, question: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    """
    Retrieves top-k matching chunks from Chroma for a user question.
    Returns list of (chunk_text, distance_score).

    IMPORTANT:
    - Chroma returns "distances" by default (smaller distance = closer match).
    - We'll still print distance so you can see ranking.
    """

    # Embed the user question
    q_embed = get_embedding(question)

    # Query Chroma using the question embedding
    results = collection.query(
        query_embeddings=[q_embed],  # list because Chroma supports batch queries
        n_results=top_k,             # top-k results
        include=["documents", "distances", "metadatas"]  # what to return
    )

    # Extract documents and distances (lists inside lists)
    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]

    # Combine into a readable list
    combined = []
    for doc_text, dist, meta in zip(docs, dists, metas):
        combined.append((doc_text, float(dist)))

    # Return list of (text, distance)
    return combined


# -------------------------------
# STEP 6: GENERATE ANSWER WITH GEMMA3:1B
# -------------------------------

def generate_answer(context_chunks: List[str], question: str) -> str:
    """
    Builds a RAG prompt:
    - Provide retrieved context
    - Ask the model to answer ONLY from context
    """

    # Join chunks into a single context string
    context = "\n\n".join(context_chunks)

    # Create a strict prompt to reduce hallucination
    prompt = f"""
You are a helpful assistant focused on Ondipudur temples.

Rules:
1) Answer ONLY using the context below.
2) If the answer is not in the context, say: "I don't have that information in the provided notes."
3) Be concise and clear.

Context:
{context}

Question:
{question}
"""

    # Call Ollama chat with gemma3:1b
    response = ollama.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    # Return generated text
    return response["message"]["content"]


# -------------------------------
# MAIN PROGRAM (END-TO-END)
# -------------------------------

def main():
    # Print app title
    print("=== ChromaDB RAG + gemma3:1b (Ollama) ===")

    # Ensure the source file exists
    if not os.path.exists(SOURCE_FILE):
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        print("Make sure Templesofondipudur.txt is in the same folder as app.py")
        return

    # Load and chunk the source file
    docs = load_and_chunk_text(SOURCE_FILE)

    # Create/load Chroma collection
    collection = get_chroma_collection()

    # Upsert documents into Chroma (adds vectors to DB)
    print(f"Indexing {len(docs)} chunks into ChromaDB...")
    upsert_documents(collection, docs)
    print("Indexing complete ✅")

    # Start interactive Q&A loop
    while True:
        # Ask the user for a question
        question = input("\nAsk a question (type exit to quit): ").strip()

        # Exit condition
        if question.lower() == "exit":
            break

        # Retrieve top-k context from Chroma
        matches = retrieve_context(collection, question, top_k=TOP_K)

        # If Chroma returns nothing (rare), handle it
        if not matches:
            print("No matching context found in ChromaDB.")
            continue

        # Show retrieved chunks (optional transparency)
        print("\n--- Retrieved Context (Top Matches) ---")
        context_texts = []
        for i, (txt, dist) in enumerate(matches, start=1):
            print(f"\nMatch #{i} | distance={dist:.4f}")
            print(txt)
            context_texts.append(txt)

        # Generate final answer using Gemma
        print("\n--- gemma3:1b Answer ---")
        answer = generate_answer(context_texts, question)
        print(answer)


# Python entry point
if __name__ == "__main__":
    main()
