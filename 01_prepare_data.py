"""Task 1: build a 200-example grammar-correction dataset and split it.

Source : synthetic. Clean sentences are generated from templates, then a
         grammar error is injected (subject-verb agreement or a/an article).
Format : CSV with columns id, error_type, input (ungrammatical), target (correct).
Split  : 140 train / 60 eval (seed 42).
"""
import itertools
import json
import random
import pandas as pd

random.seed(42)

# ---- family 1: subject-verb agreement -------------------------------------
SG = ["The student", "My brother", "The teacher", "Our neighbor", "The doctor",
      "A small child", "The manager"]
PL = ["The students", "My brothers", "The teachers", "Our neighbors",
      "The doctors", "Small children", "The managers"]
PRED = [  # (3rd-person singular, base form, rest of sentence)
    ("reads", "read", "a book every night"),
    ("walks", "walk", "to school in the morning"),
    ("plays", "play", "football after class"),
    ("writes", "write", "a letter to a friend every week"),
    ("cooks", "cook", "dinner for the family"),
    ("visits", "visit", "the library on Fridays"),
    ("drinks", "drink", "tea in the evening"),
    ("studies", "study", "hard before every exam"),
    ("watches", "watch", "the news at nine"),
    ("cleans", "clean", "the classroom after lunch"),
    ("sings", "sing", "a song at the party"),
    ("paints", "paint", "pictures on the weekend"),
]

agreement = []
for (subj, is_sg), (s_form, b_form, rest) in itertools.product(
        [(s, True) for s in SG] + [(p, False) for p in PL], PRED):
    good = s_form if is_sg else b_form
    bad = b_form if is_sg else s_form
    agreement.append((f"{subj} {bad} {rest}.", f"{subj} {good} {rest}.", "agreement"))
random.shuffle(agreement)
agreement = agreement[:150]

# ---- family 2: a / an articles ---------------------------------------------
NOUNS = [("an", "apple"), ("an", "umbrella"), ("an", "hour"), ("an", "orange"),
         ("an", "elephant"), ("an", "idea"), ("an", "egg"), ("an", "engineer"),
         ("an", "island"), ("an", "old chair"), ("a", "book"), ("a", "car"),
         ("a", "dog"), ("a", "table"), ("a", "university"), ("a", "house"),
         ("a", "window"), ("a", "phone"), ("a", "unicorn"), ("a", "banana"),
         ("a", "garden"), ("a", "river"), ("a", "ticket"), ("a", "bridge"),
         ("a", "pencil")]
TEMPL = ["She found {} {} near the park.", "He bought {} {} yesterday."]
article = []
for (art, noun), t in itertools.product(NOUNS, TEMPL):
    wrong = "a" if art == "an" else "an"
    article.append((t.format(wrong, noun), t.format(art, noun), "article"))

rows = agreement + article
random.shuffle(rows)
df = pd.DataFrame(rows, columns=["input", "target", "error_type"]).drop_duplicates("input")
df.insert(0, "id", range(len(df)))
df = df[["id", "error_type", "input", "target"]]

n_train = 140
train, evl = df.iloc[:n_train], df.iloc[n_train:]
df.to_csv("data/full.csv", index=False)
train.to_csv("data/train.csv", index=False)
evl.to_csv("data/eval.csv", index=False)

info = {"source": "synthetic template-based", "total": len(df),
        "train": len(train), "eval": len(evl),
        "error_types": df.error_type.value_counts().to_dict(),
        "preprocessing": ["template generation", "error injection",
                          "dedup", "shuffle (seed=42)", "140/60 split"]}
json.dump(info, open("data/dataset_info.json", "w"), indent=2)
print(info)
