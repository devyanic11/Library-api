from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from chat_engine import initialize_chat, chat_with_pdf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for the loaded PDF
chunks, metadata, index = None, None, None
PDF_PATH = "uploaded.pdf"

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global chunks, metadata, index
    logger.info(f"[UPLOAD] File received: {file.filename}")
    try:
        with open(PDF_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"[UPLOAD] File saved to disk as: {PDF_PATH}")
    except Exception as e:
        logger.error(f"[UPLOAD] Error saving file: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to save file: {e}"})
    try:
        logger.info("[UPLOAD] Starting PDF processing...")
        chunks, metadata, index = initialize_chat(PDF_PATH)
        logger.info(f"[UPLOAD] PDF processing complete. Chunks: {len(chunks)}, Metadata: {len(metadata)}")
        return {"message": "✅ PDF uploaded and processed successfully!"}
    except Exception as e:
        logger.error(f"[UPLOAD] Error during processing: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to process PDF: {e}"})

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global chunks, metadata, index
    if not all([chunks, metadata, index]):
        logger.info("[ASK] No PDF loaded. Rejecting question.")
        return JSONResponse(status_code=400, content={"error": "PDF not uploaded or processed yet."})
    try:
        logger.info(f"[ASK] Received question: {question}")
        answer, sources = chat_with_pdf(question, index, chunks, metadata)
        logger.info(f"[ASK] Answer generated. Returning response.")
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"[ASK] Error during answer generation: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/reset/")
async def reset_pdf():
    global chunks, metadata, index, processing, processing_error
    chunks = None
    metadata = None
    index = None
    processing = False
    processing_error = None
    return {"status": "reset"}

@app.get("/")
def root():
    return {"message": "📚 PDF Q&A API is running. Use /upload-pdf and /ask endpoints."}
