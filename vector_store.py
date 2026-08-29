"""
vector_store.py
----------------
This file handles everything related to the VECTOR DATABASE:

  Medical Documents -> Text Chunking -> Embeddings -> FAISS -> Vector Database

WHAT ARE EMBEDDINGS?
An embedding is a list of numbers (a "vector") that represents the MEANING
of a piece of text. Texts with similar meaning end up with similar vectors
(mathematically "close" to each other). We use a small, free, local model
(all-MiniLM-L6-v2) to turn text into these vectors - no API key needed for this part.

WHAT IS CHUNKING?
Our knowledge files are short, so we treat each file as one chunk. In bigger
projects, long documents are split into smaller pieces ("chunks") so that
each chunk is small enough to embed meaningfully and retrieve precisely.

WHAT IS A VECTOR DATABASE?
It's a database specialized in storing vectors and quickly finding the
"closest" vectors to a query vector. We use FAISS (Facebook AI Similarity
Search), a free local library - no server needed.

WHAT IS SIMILARITY SEARCH?
When a query comes in, we turn it into a vector too, then ask FAISS:
"which stored vectors are closest to this one?" Closeness = similarity
of meaning. The closest matches are what we retrieve.
"""

import os
import glob
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_PATH = "vectorstore/index.faiss"
META_PATH = "vectorstore/metadata.pkl"

# A small, fast, free local embedding model.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():
    """
    Read every .txt file in data/ and return a list of
    {"source": filename, "text": file_content} dictionaries.

    Each file = one chunk (they are already short and focused on one topic,
    so no further splitting is needed for this simple project).
    """
    documents = []
    for filepath in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({
            "source": os.path.basename(filepath),
            "text": text
        })
    return documents


def build_index():
    """
    STEP 1: Load documents (chunks)
    STEP 2: Convert each chunk into an embedding vector
    STEP 3: Store all vectors in a FAISS index
    STEP 4: Save the index + the original text (metadata) to disk
    """
    print("Loading documents from data/ ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [doc["text"] for doc in documents]
    print("Creating embeddings for all documents...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # FAISS needs float32 vectors
    embeddings = embeddings.astype("float32")

    # Normalize vectors so we can use inner product as cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine similarity after normalization)
    index.add(embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"Vector database built and saved to '{INDEX_PATH}' and '{META_PATH}'.")
    print(f"Total chunks indexed: {len(documents)}")


class Retriever:
    """
    Loads the saved FAISS index + metadata, and lets you search it.
    This is the "R" (Retrieval) in RAG.
    """

    def __init__(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            raise FileNotFoundError(
                "Vector database not found. Run 'python build_index.py' first."
            )
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.documents = pickle.load(f)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def search(self, query, top_k=3):
        """
        Turn the query into an embedding, then ask FAISS for the
        top_k closest document chunks (this is Similarity Search + Top-K).

        Returns a list of {"source", "text", "score"} dictionaries.
        """
        query_vector = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append({
                "source": doc["source"],
                "text": doc["text"],
                "score": float(score)
            })
        return results


if __name__ == "__main__":
    build_index()
