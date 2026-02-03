import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import easyocr
import numpy as np
from PIL import Image

app = FastAPI(title="Medical RAG - Ingestion Service")

# Initialize OCR
# Loading it outside the endpoint so it stays in memory
reader = easyocr.Reader(['fr', 'en'], gpu=False)

TEMP_DIR = Path("temp_storage")
TEMP_DIR.mkdir(exist_ok=True)

def extract_pdf_text(path: Path) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def extract_image_text(path: Path) -> str:
    results = reader.readtext(str(path))
    return " ".join([res[1] for res in results])

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    file_path = TEMP_DIR / file.filename
    
    # Save file temporarily
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    ext = file_path.suffix.lower()
    extracted_text = ""

    try:
        if ext == ".pdf":
            extracted_text = extract_pdf_text(file_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            extracted_text = extract_image_text(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if not extracted_text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted")

        return {
            "filename": file.filename,
            "text": extracted_text,
            "status": "success"
        }

    finally:
        # Cleanup: remove the file after processing
        if file_path.exists():
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
