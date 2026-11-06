"""
train.py — build the dataset, train the tiny neural network, save it.

Run:  python train.py
Outputs: model.pt (contains vocab, labels, weights, hyperparameters)
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet, tokenize, INTENTS_PATH, MODEL_PATH

BASE_DIR = Path(__file__).resolve().parent


# ---------- 1. load raw intents ----------
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

# ---------- 2. build vocabulary and label set ----------
vocab = set()
labels = []
xy = []  # (pattern_tokens, tag)

for intent in intents["intents"]:
    tag = intent["tag"]
    if tag not in labels:
        labels.append(tag)
    for pattern in intent["patterns"]:
        tokens = tokenize(pattern)
        vocab.update(tokens)
        xy.append((tokens, tag))

# sort for determinism
vocab = sorted(vocab)
labels = sorted(labels)

print(f"Vocabulary size : {len(vocab)} words")
print(f"Intents (labels): {labels}")
print(f"Training samples: {len(xy)} (patterns across all intents)")

# ---------- 3. build BoW dataset ----------
X_train = []
y_train = []
for tokens, tag in xy:
    bow = []
    for w in vocab:
        bow.append(1.0 if w in tokens else 0.0)
    X_train.append(bow)
    y_train.append(labels.index(tag))

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)


class QADataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.n = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.n


# ---------- 4. hyperparameters ----------
input_size = len(vocab)
hidden_size = 16
num_classes = len(labels)
batch_size = 8
learning_rate = 0.001
num_epochs = 300

dataset = QADataset(X_train, y_train)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ---------- 5. training loop ----------
print(f"\nTraining on {device} for {num_epochs} epochs...")
for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    correct = 0
    total = 0
    for words, labels_batch in loader:
        words = words.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * words.size(0)
        _, pred = torch.max(outputs, dim=1)
        correct += (pred == labels_batch).sum().item()
        total += words.size(0)

    if epoch % 50 == 0 or epoch == 1:
        acc = 100.0 * correct / total
        print(f"  epoch {epoch:3d}/{num_epochs} | loss={total_loss/total:.4f} | acc={acc:.1f}%")

# ---------- 6. save everything needed to load later ----------
checkpoint = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "num_classes": num_classes,
    "vocab": vocab,
    "labels": labels,
}
torch.save(checkpoint, MODEL_PATH)
print(f"\nSaved trained model to: {MODEL_PATH}")
print("You can now run:  python app.py   (FastAPI)  or   streamlit run streamlit_app.py")
