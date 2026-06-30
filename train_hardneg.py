"""
Cross-encoder retrained with HARD negatives (same-category items).

The original model (main.py) trained on random negatives only → its scores
collapsed to 0/1 (saturated), leaving the threshold nothing to work with.
Same-category negatives force the model to learn fine-grained relevance, which
spreads the score distribution and lets the ensemble + threshold actually help.

Writes to NEW paths so the original artifacts are untouched:
  cross_encoder_hn/   ce_scores_hn.npy
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

_KAGGLE_PATH = Path("/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle")
_LOCAL_PATH  = Path(__file__).parent / "kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle"
DATA_DIR = _KAGGLE_PATH if _KAGGLE_PATH.exists() else _LOCAL_PATH

MODEL_DIR   = Path(__file__).parent / "cross_encoder_hn"
SCORES_PATH = Path(__file__).parent / "ce_scores_hn.npy"

BASE_MODEL = "xlm-roberta-base"
N_TRAIN    = 80_000
EPOCHS     = 2
TRAIN_BS   = 32
INFER_BS   = 64
MAX_LEN    = 128
SEED       = 42

assert (DATA_DIR / "submission_pairs.csv").exists(), f"data not found: {DATA_DIR}"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Load test pairs + terms ────────────────────────────────────────────────
print("\nLoading test data...")
pairs      = pd.read_csv(DATA_DIR / "submission_pairs.csv")
all_terms  = pd.read_csv(DATA_DIR / "terms.csv").set_index("term_id")
term_query = all_terms["query"].to_dict()
test_terms = all_terms[all_terms.index.isin(pairs["term_id"].unique())]
needed_test_items = set(pairs["item_id"].unique())
print(f"  {len(pairs):,} test pairs | {len(test_terms):,} terms | {len(needed_test_items):,} items")

# ── 2. Item text + category builder ───────────────────────────────────────────
def make_item_text(row):
    parts = [row.get("title", ""), row.get("category", ""), row.get("brand", ""),
             row.get("gender", ""), row.get("age_group", ""), row.get("attributes", "")]
    return " ".join(str(p) for p in parts if pd.notna(p) and str(p) not in ("", "unknown", "nan"))

# ── 3. Sample training positives, then load text+category for everything we need
print("\nPreparing training data (hard negatives)...")
train_pairs  = pd.read_csv(DATA_DIR / "training_pairs.csv")
train_sample = train_pairs.sample(min(N_TRAIN, len(train_pairs)), random_state=SEED)
train_item_ids = set(train_sample["item_id"].unique())

# we need text for: training items + test items; category for the same set.
need_ids = train_item_ids | needed_test_items
item_text, item_cat = {}, {}
for chunk in pd.read_csv(DATA_DIR / "items.csv", chunksize=500_000):
    hit = chunk[chunk["item_id"].isin(need_ids)]
    for _, row in hit.iterrows():
        iid = row["item_id"]
        item_text[iid] = make_item_text(row)
        item_cat[iid]  = row.get("category", "")
    if len(item_text) >= len(need_ids):
        break
print(f"  {len(item_text):,} item texts loaded "
      f"({len(train_item_ids):,} train ∪ {len(needed_test_items):,} test)")

# build category -> item_ids pool from everything we loaded (big enough per cat)
by_cat = {}
for iid, cat in item_cat.items():
    by_cat.setdefault(cat, []).append(iid)
by_cat = {c: np.array(v) for c, v in by_cat.items()}
all_loaded = np.array(list(item_text.keys()))
print(f"  {len(by_cat):,} categories for hard-negative sampling")

# ── 4. Build (query, item, label) triples: 1 pos + 1 easy + 1 hard ────────────
rng = np.random.default_rng(SEED)
pos_set = set(zip(train_sample["term_id"], train_sample["item_id"]))

queries, item_texts, labels = [], [], []
for tid, iid in zip(train_sample["term_id"], train_sample["item_id"]):
    if tid not in term_query or iid not in item_text:
        continue
    q = str(term_query[tid])

    # positive
    queries.append(q); item_texts.append(item_text[iid]); labels.append(1.0)

    # easy negative: random product
    rid = all_loaded[rng.integers(0, len(all_loaded))]
    queries.append(q); item_texts.append(item_text.get(rid, "")); labels.append(0.0)

    # hard negative: another product in the SAME category
    pool = by_cat.get(item_cat.get(iid))
    if pool is None or len(pool) < 2:
        hid = all_loaded[rng.integers(0, len(all_loaded))]
    else:
        hid = pool[rng.integers(0, len(pool))]
        if hid == iid:
            hid = pool[rng.integers(0, len(pool))]
    queries.append(q); item_texts.append(item_text.get(hid, "")); labels.append(0.0)

print(f"  {len(queries):,} training examples "
      f"({labels.count(1.0):,} pos / {labels.count(0.0):,} neg)")

# shuffle + split
perm = np.random.default_rng(0).permutation(len(queries))
queries    = [queries[i]    for i in perm]
item_texts = [item_texts[i] for i in perm]
labels     = [labels[i]     for i in perm]

n_val = max(1000, int(0.1 * len(queries)))
val_q,  train_q  = queries[:n_val],    queries[n_val:]
val_it, train_it = item_texts[:n_val], item_texts[n_val:]
val_lb, train_lb = labels[:n_val],     labels[n_val:]

# ── 5. Train ──────────────────────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, q, it, lb): self.q, self.it, self.lb = q, it, lb
    def __len__(self):  return len(self.lb)
    def __getitem__(self, i): return self.q[i], self.it[i], self.lb[i]

if not MODEL_DIR.exists():
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def collate_fn(batch):
        q, it, lb = zip(*batch)
        enc = tokenizer(list(q), list(it), padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        return enc, torch.tensor(lb, dtype=torch.float32)

    train_dl = DataLoader(PairDataset(train_q, train_it, train_lb),
                          batch_size=TRAIN_BS, shuffle=True, collate_fn=collate_fn)
    val_dl   = DataLoader(PairDataset(val_q, val_it, val_lb),
                          batch_size=INFER_BS, shuffle=False, collate_fn=collate_fn)

    print(f"\nTraining cross-encoder (hard neg): {BASE_MODEL}  device={device}")
    print(f"  {len(train_q):,} train + {len(val_q):,} val | {EPOCHS} epochs  bs={TRAIN_BS}")

    model     = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    n_steps   = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, n_steps // 5, n_steps)
    loss_fn   = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train(); total_loss = 0.0
        bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for enc, lb in bar:
            enc = {k: v.to(device) for k, v in enc.items()}; lb = lb.to(device)
            optimizer.zero_grad()
            logits = model(**enc).logits.squeeze(-1)
            loss = loss_fn(logits, lb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval(); v_scores, v_labels = [], []
        with torch.no_grad():
            for enc, lb in val_dl:
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits.squeeze(-1)
                v_scores.extend(torch.sigmoid(logits).cpu().tolist())
                v_labels.extend(lb.tolist())
        v_scores = np.array(v_scores); v_labels = np.array(v_labels).astype(int)
        bf1, bt = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.01):
            p = (v_scores >= t).astype(int)
            if len(np.unique(p)) < 2: continue
            f = f1_score(v_labels, p, average="macro")
            if f > bf1: bf1, bt = f, float(t)
        print(f"  Epoch {epoch+1} — loss:{total_loss/len(train_dl):.4f}  "
              f"val-F1:{bf1:.4f}  thr:{bt:.2f}  "
              f"score[min/mean/max]:{v_scores.min():.2f}/{v_scores.mean():.2f}/{v_scores.max():.2f}")

        if bf1 >= best_val_f1:
            best_val_f1 = bf1
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  ✓ saved (val-F1: {best_val_f1:.4f})")
else:
    print(f"\n{MODEL_DIR.name} already exists — skipping training")

# ── 6. Score all test pairs ───────────────────────────────────────────────────
if not SCORES_PATH.exists():
    print("\nScoring all test pairs (3.36M)...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()

    q_list = [str(term_query.get(tid, "")) for tid in pairs["term_id"]]
    i_list = [item_text.get(iid, "")        for iid in pairs["item_id"]]

    scores = np.empty(len(pairs), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, len(pairs), INFER_BS), desc="Scoring", unit="batch"):
            e = min(s + INFER_BS, len(pairs))
            enc = tokenizer(q_list[s:e], i_list[s:e], padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            scores[s:e] = torch.sigmoid(model(**enc).logits.squeeze(-1)).cpu().numpy()
            if s % 200_000 == 0 and s > 0:
                torch.mps.empty_cache()
                np.save(SCORES_PATH, scores)   # checkpoint in case of crash
    np.save(SCORES_PATH, scores)
    print(f"  saved → {SCORES_PATH.name}  "
          f"(min={scores.min():.3f} mean={scores.mean():.3f} max={scores.max():.3f})")
else:
    print(f"\n{SCORES_PATH.name} already exists — skipping scoring")

print("\nDone. Next: run  CE_SCORES_PATH=ce_scores_hn.npy OUT_SUFFIX=_hn python ensemble.py")
