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

MODEL_DIR   = Path(__file__).parent / "cross_encoder_model"
SCORES_PATH = Path(__file__).parent / "ce_scores.npy"

BASE_MODEL = "xlm-roberta-base"
N_TRAIN    = 80_000
EPOCHS     = 2
TRAIN_BS   = 32
INFER_BS   = 128
MAX_LEN    = 128

assert (DATA_DIR / "submission_pairs.csv").exists(), f"veri bulunamadı: {DATA_DIR}"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Load test data ─────────────────────────────────────────────────────────
print("\nTest verisi yükleniyor...")
pairs      = pd.read_csv(DATA_DIR / "submission_pairs.csv")
test_terms = pd.read_csv(DATA_DIR / "terms.csv")
test_terms = test_terms[test_terms["term_id"].isin(pairs["term_id"].unique())].set_index("term_id")

needed_item_ids = set(pairs["item_id"].unique())
print(f"  {len(pairs):,} test çifti | {len(test_terms):,} terim | {len(needed_item_ids):,} ürün")

# ── 2. Item text builder ──────────────────────────────────────────────────────
def make_item_text(row):
    parts = [row.get("title", ""), row.get("category", ""), row.get("brand", ""),
             row.get("gender", ""), row.get("age_group", ""), row.get("attributes", "")]
    return " ".join(str(p) for p in parts if pd.notna(p) and str(p) not in ("", "unknown", "nan"))

# ── 3. Build test item text map ───────────────────────────────────────────────
print("\nTest ürünleri yükleniyor...")
test_item_text = {}
for chunk in pd.read_csv(DATA_DIR / "items.csv", chunksize=500_000):
    sub = chunk[chunk["item_id"].isin(needed_item_ids)]
    for _, row in sub.iterrows():
        test_item_text[row["item_id"]] = make_item_text(row)
    if len(test_item_text) == len(needed_item_ids):
        break
print(f"  {len(test_item_text):,} ürün metni hazır")

# ── 4. Training ───────────────────────────────────────────────────────────────
if not MODEL_DIR.exists():
    print("\nEğitim verisi hazırlanıyor...")

    train_pairs = pd.read_csv(DATA_DIR / "training_pairs.csv")
    all_terms   = pd.read_csv(DATA_DIR / "terms.csv").set_index("term_id")
    train_sample = train_pairs.sample(min(N_TRAIN, len(train_pairs)), random_state=42)

    train_item_ids = set(train_sample["item_id"].unique())
    train_item_text = {}
    for chunk in pd.read_csv(DATA_DIR / "items.csv", chunksize=500_000):
        sub = chunk[chunk["item_id"].isin(train_item_ids)]
        for _, row in sub.iterrows():
            train_item_text[row["item_id"]] = make_item_text(row)
        if len(train_item_text) == len(train_item_ids):
            break
    print(f"  {len(train_item_text):,} eğitim ürünü yüklendi")

    rng = np.random.default_rng(42)
    test_item_ids_arr = np.array(list(needed_item_ids))  # convert once

    term_query = all_terms["query"].to_dict()  # fast dict lookup

    queries, pos_texts, neg_texts = [], [], []
    for tid, iid in zip(train_sample["term_id"], train_sample["item_id"]):
        if tid not in term_query or iid not in train_item_text:
            continue
        queries.append(str(term_query[tid]))
        pos_texts.append(train_item_text[iid])
        neg_texts.append(test_item_text.get(str(rng.choice(test_item_ids_arr)), ""))

    print(f"  {len(queries):,} çift hazırlandı")

    # Flatten to (text_a, text_b, label) triples and shuffle
    all_q  = queries + queries
    all_it = pos_texts + neg_texts
    all_lb = [1.0] * len(queries) + [0.0] * len(queries)
    perm   = np.random.default_rng(0).permutation(len(all_q))
    all_q  = [all_q[i]  for i in perm]
    all_it = [all_it[i] for i in perm]
    all_lb = [all_lb[i] for i in perm]

    n_val      = max(1000, int(0.1 * len(all_q)))
    val_q,  train_q  = all_q[:n_val],  all_q[n_val:]
    val_it, train_it = all_it[:n_val], all_it[n_val:]
    val_lb, train_lb = all_lb[:n_val], all_lb[n_val:]

    class PairDataset(Dataset):
        def __init__(self, q, it, lb):
            self.q, self.it = q, it
            self.lb = lb
        def __len__(self):  return len(self.lb)
        def __getitem__(self, i): return self.q[i], self.it[i], self.lb[i]

    print(f"\nTokenizer yükleniyor: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def collate_fn(batch):
        q, it, lb = zip(*batch)
        enc = tokenizer(list(q), list(it), padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        return enc, torch.tensor(lb, dtype=torch.float32)

    train_ds = PairDataset(train_q, train_it, train_lb)
    val_ds   = PairDataset(val_q,   val_it,   val_lb)
    train_dl = DataLoader(train_ds, batch_size=TRAIN_BS, shuffle=True,  collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=INFER_BS, shuffle=False, collate_fn=collate_fn)

    print(f"\nCross-encoder eğitiliyor: {BASE_MODEL}  device={device}")
    print(f"  {len(train_ds):,} eğitim + {len(val_ds):,} val  |  {EPOCHS} epoch  bs={TRAIN_BS}")

    model     = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    n_steps   = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, n_steps // 5, n_steps)
    loss_fn   = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for batch_enc, batch_lb in bar:
            batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
            batch_lb  = batch_lb.to(device)
            optimizer.zero_grad()
            logits = model(**batch_enc).logits.squeeze(-1)
            loss   = loss_fn(logits, batch_lb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        v_logits, v_labels = [], []
        with torch.no_grad():
            for batch_enc, batch_lb in val_dl:
                batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
                logits = model(**batch_enc).logits.squeeze(-1)
                v_logits.extend(torch.sigmoid(logits).cpu().tolist())
                v_labels.extend(batch_lb.tolist())

        v_scores = np.array(v_logits)
        v_labels = np.array(v_labels).astype(int)
        best_vf1, best_vt = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.01):
            p = (v_scores >= t).astype(int)
            if len(np.unique(p)) < 2: continue
            f = f1_score(v_labels, p, average="macro")
            if f > best_vf1: best_vf1, best_vt = f, float(t)

        print(f"  Epoch {epoch+1} — loss:{total_loss/len(train_dl):.4f}  val-F1:{best_vf1:.4f}  eşik:{best_vt:.2f}")

        if best_vf1 > best_val_f1:
            best_val_f1 = best_vf1
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  ✓ Model kaydedildi (val-F1: {best_val_f1:.4f})")

# ── 5. Score all test pairs ───────────────────────────────────────────────────
if not SCORES_PATH.exists():
    print("\nCross-encoder skoru hesaplanıyor (3.36M çift)...")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()

    q_list = [str(test_terms.loc[tid, "query"]) if tid in test_terms.index else ""
              for tid in pairs["term_id"]]
    i_list = [test_item_text.get(iid, "") for iid in pairs["item_id"]]

    CHUNK     = 32   # small batches to avoid MPS OOM
    ce_scores = np.empty(len(pairs), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(pairs), CHUNK), desc="Scoring", unit="batch"):
            end = min(start + CHUNK, len(pairs))
            enc = tokenizer(q_list[start:end], i_list[start:end],
                            padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.squeeze(-1)
            ce_scores[start:end] = torch.sigmoid(logits).cpu().numpy()
            if start % 100_000 == 0 and start > 0:
                torch.mps.empty_cache()

    np.save(SCORES_PATH, ce_scores)
    print(f"  Skorlar kaydedildi → {SCORES_PATH}")
else:
    print("\nCross-encoder skorları cache'den yükleniyor...")
    ce_scores = np.load(SCORES_PATH)

print(f"  Skor dağılımı — min:{ce_scores.min():.3f} ort:{ce_scores.mean():.3f} max:{ce_scores.max():.3f}")

# ── 6. Threshold from val scores ─────────────────────────────────────────────
print("\nEşik kalibrasyonu...")

train_pairs = pd.read_csv(DATA_DIR / "training_pairs.csv")
all_terms   = pd.read_csv(DATA_DIR / "terms.csv").set_index("term_id")
val_pos     = train_pairs.sample(min(2000, len(train_pairs)), random_state=7)
val_pos     = val_pos[val_pos["item_id"].isin(test_item_text) & val_pos["term_id"].isin(all_terms.index)]

rng3 = np.random.default_rng(7)
test_item_ids_list = list(needed_item_ids)
neg_iids = [rng3.choice(test_item_ids_list) for _ in range(len(val_pos))]

val_q        = [str(all_terms.loc[tid, "query"]) for tid in val_pos["term_id"]]
val_pos_text = [test_item_text[iid] for iid in val_pos["item_id"]]
val_neg_text = [test_item_text[iid] for iid in neg_iids]

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()

def score_pairs(q, it):
    out = []
    with torch.no_grad():
        for start in range(0, len(q), INFER_BS):
            enc = tokenizer(q[start:start+INFER_BS], it[start:start+INFER_BS],
                            padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.squeeze(-1)
            out.extend(torch.sigmoid(logits).cpu().tolist())
    return np.array(out)

val_scores = np.concatenate([score_pairs(val_q, val_pos_text),
                              score_pairs(val_q, val_neg_text)])
val_labels = np.array([1] * len(val_pos) + [0] * len(val_pos))

best_f1, best_thresh = 0.0, 0.5
for thresh in np.arange(0.05, 0.95, 0.01):
    preds = (val_scores >= thresh).astype(int)
    if len(np.unique(preds)) < 2: continue
    f1 = f1_score(val_labels, preds, average="macro")
    if f1 > best_f1: best_f1, best_thresh = f1, float(thresh)

print(f"  Val eşiği: {best_thresh:.2f}  (macro-F1: {best_f1:.4f})")

# ── 7. Predict and save ───────────────────────────────────────────────────────
predictions = (ce_scores >= best_thresh).astype(np.int8)
print(f"  1 (relevant): {predictions.sum():,}  |  0 (irrelevant): {(predictions == 0).sum():,}")

submission = pd.DataFrame({"id": pairs["id"].values, "prediction": predictions})
submission.to_csv("submission_v2.csv", index=False)
print(f"\nsubmission_v2.csv kaydedildi — {len(submission):,} satır")
