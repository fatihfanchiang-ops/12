"""
model.py — the tiny neural network and inference helper.

Architecture (intent classifier, not a full LLM):
  input  : bag-of-words vector over the vocabulary built from training data
  hidden : Linear(vocab, hidden_size) -> ReLU -> Dropout
  output : Linear(hidden_size, num_intents) -> softmax over intents

The answer is looked up from the intent tag. This is the classic
chatbot / intent-classifier pattern, which is exactly what this project asks for.
"""
import json
import re
import os
from pathlib import Path

import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pt"
INTENTS_PATH = BASE_DIR / "data" / "intents.json"


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# ---------- text helpers ----------
def tokenize(sentence):
    return re.findall(r"[a-z]+", sentence.lower())


def bag_of_words(tokenized_sentence, vocab):
    """Return a BoW vector aligned with `vocab` (a list of known words)."""
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    bow = [0.0] * len(vocab)
    for w in tokenized_sentence:
        if w in word_to_idx:
            bow[word_to_idx[w]] = 1.0
    return bow


# ---------- inference singleton ----------
_state = None  # (model, vocab, labels, intents, device)


def load_model():
    global _state
    if _state is not None:
        return _state

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = NeuralNet(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents = json.load(f)

    _state = (
        model,
        checkpoint["vocab"],
        checkpoint["labels"],
        intents,
        device,
    )
    return _state


def predict(message, confidence_threshold=0.5):
    """Classify `message` and return (intent, confidence, response).
    If confidence is below the threshold, return a fallback message."""
    model, vocab, labels, intents, device = load_model()

    tokens = tokenize(message)
    bow = bag_of_words(tokens, vocab)
    x = torch.tensor(bow, dtype=torch.float32).reshape(1, -1).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        confidence = conf.item()
        label = labels[pred_idx.item()]

    if confidence < confidence_threshold:
        return {
            "intent": "unknown",
            "confidence": round(confidence, 4),
            "response": "Sorry, I don't know that yet. Ask me about my name, birthday, height, weight, shoe size, favorite subject, hobby, or future goal.",
        }

    for intent in intents["intents"]:
        if intent["tag"] == label:
            return {
                "intent": label,
                "confidence": round(confidence, 4),
                "response": intent["responses"][0],
            }

    return {
        "intent": "unknown",
        "confidence": 0.0,
        "response": "Sorry, I don't understand.",
    }
