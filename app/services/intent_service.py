import json, torch, random
from app.models.intent_model import TextTransformer
from app.utils.nlp_utils import tokenize, stem
from app.services.llm_service import llm_fallback
from app.services.recommendation_service import recommend

with open("app/data/intents.json") as f:
    intents = json.load(f)

# ── Load checkpoint ───────────────────────────────────────────────────────────
data = torch.load("data.pth", map_location="cpu")

model = TextTransformer(
    vocab_size=data["vocab_size"],
    embed_dim=data["embed_dim"],
    num_heads=data["num_heads"],
    num_classes=data["num_classes"],
    num_layers=data["num_layers"],
    max_len=data["max_len"],
)
model.load_state_dict(data["model_state"])
model.eval()

word2idx = data["word2idx"]
tags     = data["tags"]
MAX_LEN  = data["max_len"]


def encode(tokens, max_len=MAX_LEN):
    """Encode a list of tokens into a padded index sequence with [CLS] prepended."""
    indices = [word2idx["<CLS>"]]
    for t in tokens:
        s = stem(t)
        indices.append(word2idx.get(s, 0))  # 0 = <PAD> for unknown words
    indices = indices[:max_len]
    indices += [0] * (max_len - len(indices))
    return indices


def classify(msg: str):
    tokens = tokenize(msg)
    x = torch.tensor([encode(tokens)], dtype=torch.long)  # (1, MAX_LEN)

    with torch.no_grad():
        output = model(x)                              # (1, num_classes)
        probs  = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return tags[pred.item()], conf.item()


def get_intent_response(msg: str) -> str:
    tag, conf = classify(msg)
    print(f"Predicted intent: {tag}  |  confidence: {conf:.2%}")

    if conf < 0.7:
        return llm_fallback(msg)

    if tag.startswith("movie"):
        return recommend("movie", msg)

    if tag.startswith("learning"):
        return recommend("learning", msg)

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand."
