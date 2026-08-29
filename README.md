# Medical Report RAG Assistant (Educational Project)

⚠️ **This application is for educational purposes only and does not provide
medical diagnosis or replace professional medical advice. Please consult a
qualified healthcare professional.**

A beginner-friendly Retrieval-Augmented Generation (RAG) app that explains a
medical report (uploaded as an image or typed as text) using a small,
curated medical knowledge base — not the model's raw memory.

---

## 1. Project Structure

```
medical-rag/
├── app.py              # Streamlit frontend
├── rag.py               # RAG pipeline (retrieval + prompt + LLM call)
├── vector_store.py       # Chunking, embeddings, FAISS index build/search
├── build_index.py        # Script to (re)build the vector database
├── ocr.py                # Image -> text via Tesseract OCR
├── requirements.txt
├── .env.example           # Copy to .env and add your API key
├── data/                  # The medical knowledge base (15 .txt files)
└── vectorstore/           # Created automatically after you build the index
```

## 2. Setup

### a) Install Python dependencies
```bash
cd medical-rag
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### b) Install Tesseract OCR (the actual OCR program, not just the Python wrapper)
- **Windows:** download installer from https://github.com/UB-Mannheim/tesseract/wiki
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

### c) Add your API key
Copy `.env.example` to `.env` and paste your Anthropic API key:
```
ANTHROPIC_API_KEY=your-api-key-here
```
Get a key at https://console.anthropic.com/

## 3. Build the Vector Database

Run this once (and again whenever you edit files in `data/`):
```bash
python build_index.py
```
Expected output: it loads 15 documents, downloads a small embedding model
the first time, creates embeddings, and saves `vectorstore/index.faiss` +
`vectorstore/metadata.pkl`.

## 4. (Optional) Test the Retriever Alone

```python
from vector_store import Retriever
r = Retriever()
results = r.search("my hemoglobin is low, what does that mean?", top_k=3)
for res in results:
    print(res["source"], res["score"])
```
You should see `hemoglobin.txt` ranked at or near the top.

## 5. Run the App

```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually http://localhost:8501).

---

## How the Two Pipelines Work

**Image path:**
```
Medical Report Image → OCR (Tesseract) → Extracted Text (editable)
  → Retriever → Vector Database → Relevant Medical Knowledge → LLM → Explanation
```

**Text path:**
```
Medical Report Text → Retriever → Vector Database
  → Relevant Medical Knowledge → LLM → Explanation
```

Both paths **converge** at the retriever — the image is only ever used to
produce query text. The image itself is never stored in or added to the
vector database. The vector database only ever contains the 15 curated
knowledge files in `data/`.

---

## Glossary — Terms Explained Using This Project

| Term | What it means | What it does | Where it is used here |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation: give an LLM real reference text before it answers | Grounds the answer in real data instead of pure memory | The whole pipeline in `rag.py` |
| **Embeddings** | Numeric vectors representing the meaning of text | Let a computer compare meanings mathematically | Created in `vector_store.py` via `SentenceTransformer` |
| **Vectors** | A list of numbers (the embedding output) | Represents text as a point in space | Stored in the FAISS index |
| **Vector Database** | A database specialized in storing/searching vectors | Finds the closest matches to a query vector fast | FAISS index in `vectorstore/index.faiss` |
| **Chunking** | Splitting text into smaller pieces before embedding | Keeps each embedded piece focused/relevant | Each file in `data/` is treated as one chunk |
| **Retriever** | The component that searches the vector database | Takes a query, returns the most relevant chunks | The `Retriever` class in `vector_store.py` |
| **Similarity Search** | Comparing vectors to find the closest ones | Measures how "close in meaning" two texts are | `index.search()` (cosine similarity via normalized inner product) |
| **Top-K** | Returning only the K most similar results | Limits context to the most relevant info | `top_k=3` in `retriever.search()` |
| **Context** | The retrieved text inserted into the prompt | Gives the LLM facts to base its answer on | `context_text` in `rag.py` |
| **LLM** | Large Language Model — generates text | Produces the final explanation | Claude, called via the Anthropic API in `rag.py` |
| **Prompt** | The instructions + data sent to the LLM | Tells the LLM exactly what to do and with what info | `SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE` in `rag.py` |
| **Grounding** | Basing an answer on real, retrieved evidence | Reduces made-up answers | Achieved by inserting retrieved context into the prompt |
| **Hallucination** | When an LLM states something false/unsupported as fact | The problem RAG helps reduce | Guarded against via prompt instructions (rules 4–6 in `SYSTEM_PROMPT`) |
| **OCR** | Optical Character Recognition — reads text from images | Converts a report photo into editable text | `ocr.py`, using Tesseract |
| **Vision Model** | An AI model that can "see" and interpret images | Alternative to OCR (not used here, but mentioned for context) | Not used in this simple build — Tesseract OCR was chosen for simplicity |

---

## Troubleshooting

- **`TesseractNotFoundError`**: Tesseract isn't installed or not on your PATH. Reinstall it per step 2b above.
- **`FileNotFoundError: Vector database not found`**: Run `python build_index.py` first.
- **OCR text is messy**: This is normal for photos — that's why the app shows the extracted text in an editable box before analysis. Fix typos manually.
- **Slow first run**: The embedding model downloads (~80MB) the first time you run `build_index.py`.
- **API errors**: Check that `.env` exists (not `.env.example`) and contains a valid `ANTHROPIC_API_KEY`.
