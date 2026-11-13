"""
app.py — FastAPI serving the trained model.

Run:
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from model import predict

app = FastAPI(
    title="My Personal AI Model",
    description="A tiny intent-classifier neural net trained on my own Q&A dataset.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float


@app.get("/")
def health():
    return {"status": "ok", "model": "my-personal-ai v1.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = predict(req.message)
    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        confidence=result["confidence"],
    )
