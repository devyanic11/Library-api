# PDF Q&A API (FastAPI + Ollama + Local Embeddings)

This project provides a REST API for chatting with the contents of a PDF document. It uses local embeddings (Sentence Transformers), FAISS for vector search, and Ollama (with Llama 2/3) for language model responses. The API is built with FastAPI for modern, async, and production-ready serving.

---

## Features
- Upload a PDF and process its contents
- Ask questions about the uploaded PDF and get answers with source citations
- Uses local LLMs via Ollama (no OpenAI/Groq API keys required)
- Fast retrieval with FAISS and local embeddings

---

## Requirements
- Python 3.8+
- [Ollama](https://ollama.com/) (running locally)
- [Llama 2/3 model](https://ollama.com/library/llama2) pulled via Ollama

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. (Recommended) Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install fastapi uvicorn PyMuPDF faiss-cpu numpy requests sentence-transformers python-multipart
```

### 4. Install and Start Ollama
- Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)
- Start the Ollama server (it usually starts automatically, or run `ollama serve` in a terminal)

### 5. Pull the Llama Model
```bash
ollama pull llama3
```

---

## Running the API Server

Start the FastAPI server with Uvicorn:
```bash
uvicorn main:app --reload
```
You should see output like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```
The server will run at `http://localhost:8000` by default.

---

## Example API Usage

### 1. Upload a PDF
Replace `your_document.pdf` with the path to your PDF file.
```bash
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/upload-pdf/
```
Expected response:
```json
{"message": "✅ PDF uploaded and processed successfully!"}
```

### 2. Ask a Question
```bash
curl -X POST -F "question=What is the PDF about?" http://localhost:8000/ask/
```
Expected response:
```json
{
  "question": "What is the main topic of this document?",
  "answer": "The answer from the model...",
  "sources": ["[1] → Page 1", "[2] → Page 3"]
}
```

---

## Notes
- The API keeps only one PDF in memory at a time. Uploading a new PDF replaces the previous one.
- The Ollama server must be running and accessible at `http://localhost:11434`.
- The first request may take longer as embeddings and the FAISS index are built.
- Temporary files are stored in the project directory as `uploaded.pdf` during processing.

---

## Troubleshooting
- **No output in console:** Make sure you run `uvicorn main:app --reload` and see the Uvicorn startup message.
- **Ollama errors:** Ensure Ollama is running and the Llama model is pulled.
- **Missing dependencies:** Install all Python requirements as shown above.
- **FAISS errors:** If you see errors about FAISS, ensure you installed `faiss-cpu` (not just `faiss`).
- **ImportError for huggingface_hub:** Downgrade with `pip install "huggingface_hub==0.15.1"`

---

## License
MIT 