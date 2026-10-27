"""Task 3 (steps 2-5): blind A/B human annotation in the terminal.

Evaluators see two responses in random order (source hidden).
Criteria: prefer the response that (1) fixes the grammar error,
(2) keeps the original meaning, (3) adds no extra/odd words.
Progress is saved after every answer, so you can quit (q) and resume.
Annotators are identified only by a short anonymous label (e.g. "A1").

Run:  uv run 04_collect_preferences.py --annotator A1
"""
import argparse
import json
import os
import random
import pandas as pd

OUT_CSV = "outputs/preference_dataset.csv"
OUT_JSON = "outputs/preference_dataset.json"
COLS = ["pair_id", "annotator", "input", "response_a", "response_b",
        "preferred", "chosen", "rejected", "chosen_source"]

ap = argparse.ArgumentParser()
ap.add_argument("--annotator", default="A1", help="anonymous label, no real names")
args = ap.parse_args()

pairs = json.load(open("data/candidates.json"))
done = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else pd.DataFrame(columns=COLS)
seen = set(zip(done.pair_id, done.annotator))
todo = [p for p in pairs if (p["pair_id"], args.annotator) not in seen]
random.Random(hash(args.annotator) % 1000).shuffle(todo)

print(__doc__)
print(f"{len(done)} comparisons already stored. Answer: a / b / t (tie) / q (quit)\n")

for p in todo:
    flip = random.random() < 0.5                      # hide the source
    a, b = (p["sampled"], p["greedy"]) if flip else (p["greedy"], p["sampled"])
    src = {"A": "sampled" if flip else "greedy", "B": "greedy" if flip else "sampled"}
    print(f"[{len(done) + 1}] Input : {p['input']}\n     A : {a}\n     B : {b}")
    ans = ""
    while ans not in {"a", "b", "t", "q"}:
        ans = input("Better? (a/b/t/q): ").strip().lower()
    if ans == "q":
        break
    if ans == "t":
        pref, chosen, rejected, csrc = "tie", "", "", ""
    else:
        pref = ans.upper()
        chosen, rejected = (a, b) if ans == "a" else (b, a)
        csrc = src[pref]
    row = dict(pair_id=p["pair_id"], annotator=args.annotator, input=p["input"],
               response_a=a, response_b=b, preferred=pref, chosen=chosen,
               rejected=rejected, chosen_source=csrc)
    done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
    done.to_csv(OUT_CSV, index=False)
    done.to_json(OUT_JSON, orient="records", indent=2, force_ascii=False)
    print()

print(f"\nTotal stored comparisons: {len(done)} "
      f"({'OK' if len(done) >= 100 else 'need ' + str(100 - len(done)) + ' more'})")
