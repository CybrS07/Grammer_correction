# Simplified RLHF – Grammar Correction (Tasks 1-3)

Setup (uv):
```bash
uv sync
uv run 01_prepare_data.py          # Task 1  -> data/
uv run 02_baseline.py              # Task 2  -> outputs/baseline_*.csv/json
uv run 03_generate_candidates.py   # Task 3  -> data/candidates.json
uv run 04_collect_preferences.py --annotator A1   # Task 3 -> outputs/preference_dataset.csv/json
```
Task 3 needs >= 100 comparisons. Run step 4 several times with different
labels (A1, A2, ...) so friends can help; the script resumes automatically.

Dataset: synthetic, template-generated grammar errors (200 examples, 140 train / 60 eval).
Model: google/flan-t5-small. Metrics: exact match, token F1, char similarity.
