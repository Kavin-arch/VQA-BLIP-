"""
FastAPI Backend for Image Captioning + VQA
---------------------------------------------
Loads BLIP models ONCE at startup, then exposes two endpoints:
  POST /caption  -> upload an image, get a description back
  POST /vqa      -> upload an image + a question, get an answer back

Run with:
    uvicorn main:app --reload --port 8000
"""

import io
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)

app = FastAPI(title="Image Captioning + VQA API")

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[startup] Using device: {device}")

# ---------------------------------------------------------------
# Load both models ONCE when the server starts (not per-request!)
# This is the key benefit of a backend: no reloading a huge model
# every time someone asks a question.
# ---------------------------------------------------------------
print("[startup] Loading captioning model...")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

print("[startup] Loading VQA model...")
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

print("[startup] Models loaded. Ready to serve requests.")


def _load_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")


@app.get("/")
def health_check():
    return {"status": "ok", "device": device}


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    """Upload an image, get back a generated caption."""
    image_bytes = await file.read()
    image = _load_image(image_bytes)

    inputs = caption_processor(image, return_tensors="pt").to(device)
    output = caption_model.generate(**inputs, max_length=50)
    caption = caption_processor.decode(output[0], skip_special_tokens=True)

    return {"caption": caption}


@app.post("/vqa")
async def visual_question_answering(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """Upload an image + a question, get back an answer."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    image_bytes = await file.read()
    image = _load_image(image_bytes)

    inputs = vqa_processor(image, question, return_tensors="pt").to(device)
    output = vqa_model.generate(**inputs, max_length=20)
    answer = vqa_processor.decode(output[0], skip_special_tokens=True)

    return {"question": question, "answer": answer}
