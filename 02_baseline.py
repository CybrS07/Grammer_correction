"""Task 2: baseline model outputs + metrics.

Model: google/flan-t5-small, greedy decoding, zero-shot prompt.
Metrics (reference-based, no extra deps):
  exact_match  - output == target
  token_f1     - word-level F1 vs target
  char_sim     - difflib similarity ratio vs target
"""
import json
from collections import Counter
from difflib import SequenceMatcher
import pandas as pd
from common import load_model, generate, set_seed


def token_f1(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def score(df, col):
    return {
        "exact_match": float((df[col].str.strip() == df.target).mean()),
        "token_f1": float(sum(token_f1(p, t) for p, t in zip(df[col], df.target)) / len(df)),
        "char_sim": float(sum(SequenceMatcher(None, p, t).ratio()
                              for p, t in zip(df[col], df.target)) / len(df)),
    }


if __name__ == "__main__":
    set_seed()
    tok, model, device = load_model()
    train = pd.read_csv("data/train.csv")
    evl = pd.read_csv("data/eval.csv")

    for df in (train, evl):
        df["baseline_output"] = generate(tok, model, device, df.input.tolist(),
                                         do_sample=False, num_beams=1)

    train.to_csv("outputs/baseline_train_outputs.csv", index=False)
    evl.to_csv("outputs/baseline_eval_outputs.csv", index=False)

    metrics = {
        "model": "google/flan-t5-small",
        "no_change_input_vs_target": score(evl.assign(baseline_output=evl.input), "baseline_output"),
        "baseline_eval": score(evl, "baseline_output"),
    }
    json.dump(metrics, open("outputs/baseline_metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
