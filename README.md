# AI Equity Analyst — Setup & Run Guide

A RAG-based Q&A tool for financial reports (PDF), with conversational memory and multi-document comparison.

## 1. Prerequisites

- Python 3.9 or higher installed
- A Google API key with access to Gemini (get one at [Google AI Studio](https://aistudio.google.com/app/apikey))

## 2. Clone / place the project

Put `equity_analyst_app.py` in its own project folder, e.g.:

```
equity-analyst/
└── equity_analyst_app.py
```

## 3. Create and activate a virtual environment (recommended)

```bash
cd equity-analyst
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

## 4. Install dependencies

Create a `requirements.txt` file in the project folder with:

```
streamlit
langchain
langchain-community
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-google-genai
langchain-chroma
pypdf
sentence-transformers
```

Then install:

```bash
pip install -r requirements.txt
```

> First install may take a few minutes — `sentence-transformers` downloads the local embedding model (`all-MiniLM-L6-v2`) dependencies.

## 5. Add your API key (never hardcode it)

Create a `.streamlit` folder inside your project, and inside it a `secrets.toml` file:

```
equity-analyst/
├── equity_analyst_app.py
├── requirements.txt
└── .streamlit/
    └── secrets.toml
```

**`.streamlit/secrets.toml`:**

```toml
GOOGLE_API_KEY = "your-actual-api-key-here"
```

**Important:** add `.streamlit/secrets.toml` to `.gitignore` so the key never gets committed:

```bash
echo ".streamlit/secrets.toml" >> .gitignore
echo "venv/" >> .gitignore
```

## 6. Run the app

```bash
streamlit run equity_analyst_app.py
```

This opens the app automatically in your browser, usually at `http://localhost:8501`.

## 7. Using the app

1. Upload one or more financial report PDFs (10-K, 10-Q, earnings releases, etc.) using the file uploader.
2. Wait for the "document(s) ready to query" success message — this means chunking + embedding finished.
3. Ask questions in the chat box at the bottom, e.g.:
   - "What was total revenue?"
   - "How does that compare to last year?" (follow-up — uses conversational memory)
   - "Compare revenue growth between these two reports" (if 2+ documents uploaded)
4. Use **🗑️ Clear conversation** to reset the chat while keeping documents loaded.
5. Use **🧹 Clear all documents** to remove all uploaded documents and start fresh.

## 8. Common issues

| Problem                                     | Fix                                                                                                                            |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `GOOGLE_API_KEY not found` error on startup | Check `.streamlit/secrets.toml` exists, is correctly formatted, and is in the same folder you're running `streamlit run` from. |
| "No extractable text found" on upload       | The PDF is likely scanned/image-only. Run it through OCR first, or use a text-based PDF.                                       |
| Slow first upload                           | Normal — the embedding model loads into memory on first use. Subsequent uploads in the same session are faster.                |
| `ModuleNotFoundError`                       | Re-run `pip install -r requirements.txt` inside the activated virtual environment.                                             |

## 9. Stopping the app

Press `Ctrl+C` in the terminal where `streamlit run` is running.
