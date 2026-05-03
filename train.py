# train.py
import json, torch, numpy as np
from app.utils.nlp_utils import tokenize, stem, bag_of_words
from app.models.intent_model import NeuralNet

with open('app/data/intents.json') as f:
    intents = json.load(f)

all_words, tags, xy = [], [], []

for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, intent['tag']))


all_words = sorted(set([stem(w) for w in all_words]))
tags = sorted(set(tags))

print(f"Number of unique words: {len(all_words)}")
print(f"Unique words: {all_words}")
print(f"Number of tags: {len(tags)}")
print(f"Unique tags: {tags}")


X_train, y_train = [], []
for (pattern, tag) in xy:
    X_train.append(bag_of_words(pattern, all_words))
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

model = NeuralNet(len(X_train[0]), 8, len(tags))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    for i in range(len(X_train)):
        x = torch.from_numpy(X_train[i]).float()
        y = torch.tensor(y_train[i])
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save({
    "model_state": model.state_dict(),
    "input_size": len(X_train[0]),
    "hidden_size": 8,
    "output_size": len(tags),
    "all_words": all_words,
    "tags": tags
}, "data.pth")

print("Training complete")
