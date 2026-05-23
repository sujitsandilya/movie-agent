import json
import torch
import numpy as np
from app.utils.nlp_utils import tokenize, stem
from app.models.intent_model import TextTransformer

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_DIM  = 64
NUM_HEADS  = 4
NUM_LAYERS = 2
MAX_LEN    = 32
DROPOUT    = 0.1
EPOCHS     = 300
LR         = 0.001
# ─────────────────────────────────────────────────────────────────────────────

with open('app/data/intents.json') as f:
    intents = json.load(f)

all_words, tags, xy = [], [], []

for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, intent['tag']))

print(f"Before stemming:")
print(f"  Unique words : {len(set(all_words))}")
print(f"  Tags         : {tags}")

all_words = sorted(set([stem(w) for w in all_words]))
tags = sorted(set(tags))

print(f"\nAfter stemming:")
print(f"  Unique words : {len(all_words)}")
print(f"  Tags         : {tags}")

# ── Build vocabulary (reserve 0 for <PAD>, add [CLS] token) ──────────────────
word2idx = {"<PAD>": 0, "<CLS>": 1}
for w in all_words:
    word2idx[w] = len(word2idx)

vocab_size = len(word2idx)


def encode(tokens, max_len=MAX_LEN):
    """Tokenise → stem → index. Prepend [CLS], pad/truncate to max_len."""
    indices = [word2idx["<CLS>"]]
    for t in tokens:
        s = stem(t)
        indices.append(word2idx.get(s, 0))  # 0 = <PAD> for unknown words
    indices = indices[:max_len]
    # Pad to max_len
    indices += [0] * (max_len - len(indices))
    return indices


# ── Build training tensors ────────────────────────────────────────────────────
X_train, y_train = [], []
for (pattern_tokens, tag) in xy:
    X_train.append(encode(pattern_tokens))
    y_train.append(tags.index(tag))

X_train = torch.tensor(X_train, dtype=torch.long)    # (N, MAX_LEN)
y_train = torch.tensor(y_train, dtype=torch.long)    # (N,)

print(f"\nTraining data shape: {X_train.shape}")

# ── Model, loss, optimiser ───────────────────────────────────────────────────
model = TextTransformer(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_classes=len(tags),
    num_layers=NUM_LAYERS,
    max_len=MAX_LEN,
    dropout=DROPOUT,
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nTraining …")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)           # forward pass on full batch
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        preds = output.argmax(dim=1)
        acc = (preds == y_train).float().mean().item()
        print(f"  Epoch [{epoch+1:>3}/{EPOCHS}]  loss={loss.item():.4f}  acc={acc:.2%}")

# ── Save checkpoint ───────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "vocab_size":  vocab_size,
    "embed_dim":   EMBED_DIM,
    "num_heads":   NUM_HEADS,
    "num_layers":  NUM_LAYERS,
    "num_classes": len(tags),
    "max_len":     MAX_LEN,
    "word2idx":    word2idx,
    "tags":        tags,
}, "data.pth")

print("\nTraining complete — model saved to data.pth")
