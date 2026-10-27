"""Shared helpers: model loading and batched generation."""
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"   # small pretrained model (~80M params)
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)


def make_prompt(text: str) -> str:
    return f"Correct the grammar of this sentence: {text}"


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device).eval()
    return tok, model, device


@torch.no_grad()
def generate(tok, model, device, texts, batch_size=16, **gen_kwargs):
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = [make_prompt(t) for t in texts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(device)
        ids = model.generate(**enc, max_new_tokens=64, **gen_kwargs)
        outs += [s.strip() for s in tok.batch_decode(ids, skip_special_tokens=True)]
    return outs
