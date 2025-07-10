import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Global embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Step 1: Extract Text from PDF ---
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append((i + 1, text))
    return pages

# --- Step 2: Chunk Text ---
def chunk_text_with_metadata(pages, chunk_size=500, overlap=50):
    chunks = []
    metadata = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            metadata.append({"page": page_num, "start": start, "end": end})
            start += chunk_size - overlap
    return chunks, metadata

# --- Step 3: Build FAISS Index ---
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# --- Step 4: Retrieve Relevant Chunks ---
def get_relevant_chunks_with_sources(query, index, chunks, metadata, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    retrieved = []
    for i in I[0]:
        retrieved.append({"text": chunks[i], "meta": metadata[i]})
    return retrieved

# --- Step 5: Format Prompt with Sources ---
def format_prompt_with_sources(retrieved_chunks, question):
    context = ""
    sources = []
    for i, chunk in enumerate(retrieved_chunks):
        tag = f"[{i+1}]"
        context += f"{tag} {chunk['text'].strip()}\n"
        sources.append(f"{tag} → Page {chunk['meta']['page']}")
    prompt = f"""
Use the following context to answer the question. Cite sources as [1], [2], etc.

{context}

Q: {question}
A:
"""
    return prompt, sources

# --- Step 6: Ask LLaMA via Ollama HTTP API ---
def ask_llama_ollama(prompt, model="llama3"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        raise Exception(f"Ollama API error: {response.text}")

# --- Step 7: Chat Handler ---
def initialize_chat(pdf_path):
    pages = extract_text_from_pdf(pdf_path)
    chunks, metadata = chunk_text_with_metadata(pages)
    index, _ = build_faiss_index(chunks)
    return chunks, metadata, index

def chat_with_pdf(question, index, chunks, metadata, top_k=3):
    retrieved = get_relevant_chunks_with_sources(question, index, chunks, metadata, k=top_k)
    prompt, sources = format_prompt_with_sources(retrieved, question)
    answer = ask_llama_ollama(prompt)
    return answer, sources

# --- Main ---
if __name__ == "__main__":
    chunks, metadata, index = initialize_chat("testdoc.pdf")
    print("✅ PDF Loaded. Ask anything! Type 'exit' to quit.\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        try:
            answer, sources = chat_with_pdf(question, index, chunks, metadata)
            print("\n🤖 Answer:", answer)
            print("📚 Sources:", ", ".join(sources), "\n")
        except Exception as e:
            print(f"⚠️ Error: {e}")
