from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
from typing import List, Dict
from io import BytesIO
from PIL import Image
import pytesseract

# Load env vars
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Init Groq client
client = Groq(api_key=groq_api_key)

# FastAPI app
app = FastAPI()


# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod use frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SCHEMA ----------------
class QueryRequest(BaseModel):
    query: str
    system_prompt: str = """You are Frank — a clever, friendly tutor who explains things like a real teacher.
Speak clearly, break concepts into digestible parts, highlight key terms, and use an encouraging tone.
No emojis, no asterisks, no fancy symbols."""
    model: str = "openai/gpt-oss-20b"
    history: list[dict] = []
    current_topic: str = ""
    is_detailed: bool = False


# ---------------- HELPERS ----------------
def format_detailed_response(response: str) -> str:
    lines = response.split('\n')
    formatted = []
    for line in lines:
        if line.strip().startswith('-'):
            formatted.append(line)
        elif ':' in line:
            key, value = line.split(':', 1)
            formatted.append(f"**{key.strip()}:** {value.strip()}")
        else:
            formatted.append(line)
    return '\n'.join(formatted)


def format_concise_response(response: str) -> str:
    response = ' '.join(response.split())
    trigger_keywords = ['steps', 'key points', 'tips', 'benefits']
    if any(response.lower().startswith(k) for k in trigger_keywords) and ',' in response:
        points = [point.strip() for point in response.split(',')]
        return '<br>• ' + '<br>• '.join(points)
    return response


# ---------------- TEXT Q&A ----------------
@app.post("/ask")
async def ask_bot(request: QueryRequest):
    try:
        wants_detail = any(word in request.query.lower()
                           for word in ['explain', 'detail', 'elaborate', 'how', 'why'])

        base_prompt = request.system_prompt
        if "exam" in request.query.lower() or "brief" in request.query.lower():
            base_prompt += "\nProvide a concise, exam-focused response."
        elif wants_detail:
            base_prompt += "\nProvide a detailed explanation with examples."

        messages = [{"role": "system", "content": base_prompt}]
        messages.extend(request.history)
        messages.append({"role": "user", "content": request.query})

        chat_completion = client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=0.75,
            max_completion_tokens=wants_detail and 500 or 150,
            top_p=0.9,
            stream=False
        )

        response = chat_completion.choices[0].message.content

        if wants_detail:
            response = format_detailed_response(response)
        else:
            response = format_concise_response(response)

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- TOPIC EXTRACTION ----------------
@app.post("/extract-topic")
async def extract_topic(request: QueryRequest):
    try:
        topic_prompt = f"""Extract the **single most relevant topic word**.
Message: "{request.query}"
Topic:"""

        messages = [{"role": "user", "content": topic_prompt}]
        response = client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=10
        )

        topic = response.choices[0].message.content.strip().split()[0].capitalize()
        return {"topic": topic}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- IMAGE ANALYSIS ----------------
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read image + OCR
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        ocr_text = pytesseract.image_to_string(image)

        if not ocr_text.strip():
            return JSONResponse({"error": "No readable text found in the image."}, status_code=400)

        # Send OCR text to Groq tutor
        tutor_completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are Frank, a friendly tutor who explains text clearly for students."},
                {"role": "user", "content": f"Explain this extracted text for a student: {ocr_text}"}
            ],
            temperature=0.75,
            max_completion_tokens=600,
            top_p=0.9
        )

        explanation = tutor_completion.choices[0].message.content

        return JSONResponse(content={
            "ocr_text": ocr_text,
            "tutor_explanation": explanation
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------- ROOT ----------------
@app.get("/")
async def root():
    return {"message": "Frank Tutor API is running"}
