"""Task 3 (steps 1): two candidate responses per training input.

Candidate 1: greedy decoding        (the baseline output)
Candidate 2: temperature sampling   (re-sampled until it differs from #1)
Inputs where both come out identical are dropped (nothing to compare).
"""
import json
import pandas as pd
from common import load_model, generate, set_seed

set_seed()
tok, model, device = load_model()
train = pd.read_csv("data/train.csv")
texts = train.input.tolist()

greedy = generate(tok, model, device, texts, do_sample=False, num_beams=1)
sampled = generate(tok, model, device, texts, do_sample=True, top_p=0.95, temperature=1.3)

# retry identical pairs up to 8 times
for _ in range(8):
    same = [i for i in range(len(texts)) if sampled[i] == greedy[i]]
    if not same:
        break
    redo = generate(tok, model, device, [texts[i] for i in same],
                    do_sample=True, top_p=0.95, temperature=1.5)
    for i, r in zip(same, redo):
        sampled[i] = r

pairs = [{"pair_id": int(r.id), "input": r.input,
          "greedy": g, "sampled": s}
         for r, g, s in zip(train.itertuples(), greedy, sampled) if g != s]
json.dump(pairs, open("data/candidates.json", "w"), indent=2, ensure_ascii=False)
print(f"{len(pairs)} usable pairs (need >= 100 annotated comparisons)")
