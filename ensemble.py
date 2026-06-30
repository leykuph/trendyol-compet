"""
Ensemble: Cross-encoder (ce_scores.npy) + LightGBM lexical features.

Rank-normalizes both score sources to [0,1] and blends them, then emits a GRID
of candidate submissions at several global positive-rates. Macro-F1 peaks when
the predicted positive-rate is close to the true one, so submitting a few of
these tomorrow empirically finds the best operating point.

Env overrides (for the overnight re-run on the hard-negative model):
  CE_SCORES_PATH=ce_scores_hn.npy   OUT_SUFFIX=_hn
"""
import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
_KAGGLE_PATH = Path("/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle")
_LOCAL_PATH  = Path(__file__).parent / "kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle"
DATA_DIR   = _KAGGLE_PATH if _KAGGLE_PATH.exists() else _LOCAL_PATH
HERE       = Path(__file__).parent
CE_SCORES  = HERE / os.environ.get("CE_SCORES_PATH", "ce_scores.npy")
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")

assert CE_SCORES.exists(), f"{CE_SCORES.name} not found — run main.py / train_hardneg.py first"
assert (DATA_DIR / "submission_pairs.csv").exists(), f"data not found: {DATA_DIR}"

SEED        = 42
N_EASY      = 1
N_HARD      = 1
ALPHA       = 0.5                       # CE weight; (1-ALPHA) for LightGBM
# friend's log shows the optimal positive-rate is ~24% (0.293 already drops to 0.68)
POS_RATES   = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]   # candidate global positive rates

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading data...  (CE={CE_SCORES.name}, suffix='{OUT_SUFFIX or 'none'}')")
terms = pd.read_csv(DATA_DIR / "terms.csv")
items = pd.read_csv(DATA_DIR / "items.csv")
train = pd.read_csv(DATA_DIR / "training_pairs.csv")
sub   = pd.read_csv(DATA_DIR / "submission_pairs.csv")
print(f"  {len(sub):,} test pairs | {len(train):,} training positives")

# ── Turkish tokenizer ─────────────────────────────────────────────────────────
_tok = re.compile(r"[a-zçğıöşü0-9]+")
def tokenize(text):
    if not isinstance(text, str): return set()
    return set(_tok.findall(text.lower()))

# ── Lexical features ──────────────────────────────────────────────────────────
def build_features(pairs: pd.DataFrame) -> pd.DataFrame:
    t    = terms[terms.term_id.isin(pairs.term_id.unique())]
    qtok = {r.term_id: tokenize(r.query) for r in t.itertuples()}

    it = items[items.item_id.isin(pairs.item_id.unique())]
    title_tok, cat_tok, attr_tok, all_tok, brand_str = {}, {}, {}, {}, {}
    for r in it.itertuples():
        tt = tokenize(r.title); ct = tokenize(r.category)
        at = tokenize(r.attributes); bt = tokenize(r.brand)
        title_tok[r.item_id] = tt;  cat_tok[r.item_id] = ct
        attr_tok[r.item_id]  = at;  all_tok[r.item_id] = tt | ct | at | bt
        brand_str[r.item_id] = r.brand if isinstance(r.brand, str) else ""

    rows, empty = [], set()
    for tid, iid in zip(pairs.term_id.values, pairs.item_id.values):
        q  = qtok.get(tid, empty); qn = len(q) or 1
        tt = title_tok.get(iid, empty); cat = cat_tok.get(iid, empty)
        at = attr_tok.get(iid, empty);  alltok = all_tok.get(iid, empty)
        brand = brand_str.get(iid, "")
        it_ = len(q & tt); ut = len(q | tt) or 1
        rows.append((
            qn, len(tt), it_,
            it_ / qn,
            len(q & cat)    / qn,
            len(q & at)     / qn,
            len(q & alltok) / qn,
            it_ / ut,
            qn - len(q & alltok),
            1 if brand and brand.lower() in {w.lower() for w in q} else 0,
        ))
    cols = ["q_n","title_n","inter_title","cov_title","cov_cat",
            "cov_attr","cov_all","jaccard_title","n_unmatched","brand_in_q"]
    return pd.DataFrame(rows, columns=cols, dtype=np.float32)

# ── TF-IDF cosine feature ─────────────────────────────────────────────────────
def fit_tfidf():
    item_text = (items.title.fillna("") + " " +
                 items.category.fillna("") + " " +
                 items.brand.fillna(""))
    vec = TfidfVectorizer(min_df=2, max_features=120_000, dtype=np.float32)
    M   = normalize(vec.fit_transform(item_text))
    item_row = {iid: i for i, iid in enumerate(items.item_id)}
    return vec, M, item_row

def cosine_feature(pairs: pd.DataFrame, vec, M, item_row) -> np.ndarray:
    uniq  = pairs.term_id.unique()
    tq    = terms.set_index("term_id")["query"].reindex(uniq).fillna("")
    Q     = normalize(vec.transform(tq.values))
    t_row = {t: i for i, t in enumerate(uniq)}
    q_idx = pairs.term_id.map(t_row).to_numpy()
    i_idx = pairs.item_id.map(item_row)
    valid = i_idx.notna().to_numpy()
    i_idx = i_idx.fillna(0).astype(int).to_numpy()
    cos   = np.asarray(Q[q_idx].multiply(M[i_idx]).sum(axis=1)).ravel()
    cos[~valid] = 0.0
    return cos.astype(np.float32)

# ── Negative sampling (easy random + hard same-category) ──────────────────────
def build_training_set() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    pos = train[["term_id","item_id"]].drop_duplicates().reset_index(drop=True)
    pos_set = set(zip(pos.term_id, pos.item_id))
    all_ids = items.item_id.to_numpy()
    cat_of  = dict(zip(items.item_id, items.category))
    by_cat  = {c: g.item_id.to_numpy() for c, g in items.groupby("category", sort=False)}

    neg_t, neg_i = [], []
    def add(tid, c):
        if (tid, c) not in pos_set:
            neg_t.append(tid); neg_i.append(c)

    for _ in range(N_EASY):
        for tid, c in zip(pos.term_id.values,
                          all_ids[rng.integers(0, len(all_ids), size=len(pos))]):
            add(tid, c)

    for _ in range(N_HARD):
        for tid, iid in zip(pos.term_id.values, pos.item_id.values):
            pool = by_cat.get(cat_of.get(iid))
            if pool is None or len(pool) < 2:
                c = all_ids[rng.integers(0, len(all_ids))]
            else:
                c = pool[rng.integers(0, len(pool))]
                if c == iid: c = pool[rng.integers(0, len(pool))]
            add(tid, c)

    neg = pd.DataFrame({"term_id": neg_t, "item_id": neg_i}).drop_duplicates()
    neg["label"] = 0
    out = pos.copy(); out["label"] = 1
    return pd.concat([out, neg], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

def group_split(df: pd.DataFrame, val_frac=0.2):
    rng = np.random.default_rng(SEED)
    groups = np.asarray(df.term_id.unique(), dtype=object)
    groups = groups[rng.permutation(len(groups))]
    val_g  = set(groups[:int(len(groups) * val_frac)])
    mask   = df.term_id.isin(val_g)
    return df[~mask].copy(), df[mask].copy()

def best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), average="macro")
        if f1 > best_f1: best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def rank_norm(arr: np.ndarray) -> np.ndarray:
    """Percentile-rank to [0,1] — makes two different score scales comparable."""
    return (rankdata(arr, method="average") - 1) / (len(arr) - 1)

# ── Train LightGBM ────────────────────────────────────────────────────────────
print("\nBuilding negatives (easy random + hard same-category)...")
ds = build_training_set()
print(f"  {ds.label.value_counts().to_dict()}")

print("Fitting TF-IDF on catalog...")
vec, M, item_row = fit_tfidf()

tr, va = group_split(ds)
print(f"Extracting train/val features (train={len(tr):,}, val={len(va):,})...")
Xtr = build_features(tr); Xtr["tfidf_cos"] = cosine_feature(tr, vec, M, item_row)
Xva = build_features(va); Xva["tfidf_cos"] = cosine_feature(va, vec, M, item_row)
ytr, yva = tr.label.values, va.label.values

print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
)
lgb_model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
thr_lgb, f1_lgb = best_threshold(yva, lgb_model.predict_proba(Xva)[:, 1])
print(f"  LightGBM val macro-F1={f1_lgb:.4f}  (val threshold={thr_lgb:.2f})")

# ── Score the test pairs ──────────────────────────────────────────────────────
print("\nExtracting test features (3.36M pairs)...")
Xsub = build_features(sub); Xsub["tfidf_cos"] = cosine_feature(sub, vec, M, item_row)
lgbm_probs = lgb_model.predict_proba(Xsub)[:, 1]

print("Loading cross-encoder scores...")
ce_probs = np.load(CE_SCORES).astype(np.float64)
if ce_probs.min() < 0 or ce_probs.max() > 1:
    ce_probs = 1 / (1 + np.exp(-ce_probs))
print(f"  CE   min={ce_probs.min():.3f} mean={ce_probs.mean():.3f} max={ce_probs.max():.3f}")
print(f"  LGBM min={lgbm_probs.min():.3f} mean={lgbm_probs.mean():.3f} max={lgbm_probs.max():.3f}")

# ── Rank-blend ────────────────────────────────────────────────────────────────
combined = ALPHA * rank_norm(ce_probs) + (1 - ALPHA) * rank_norm(lgbm_probs)

# persist raw arrays so any future positive-rate is instant (no recompute)
np.save(HERE / f"lgbm_probs{OUT_SUFFIX}.npy", lgbm_probs.astype(np.float32))
np.save(HERE / f"combined{OUT_SUFFIX}.npy",   combined.astype(np.float32))

# ── Emit a grid of submissions at fixed global positive-rates ─────────────────
def save_grid(score: np.ndarray, tag: str):
    order = np.argsort(-score)            # highest score first
    n = len(score)
    for q in POS_RATES:
        k = int(round(q * n))
        preds = np.zeros(n, dtype=np.int8)
        preds[order[:k]] = 1
        out = pd.DataFrame({"id": sub["id"].values, "prediction": preds})
        name = f"submission_{tag}_p{int(q*100)}{OUT_SUFFIX}.csv"
        out.to_csv(HERE / name, index=False)
        print(f"  {name}: pos={preds.sum():,} ({q:.0%})")

print("\nWriting candidate submissions (blend, varying positive-rate):")
save_grid(combined, "blend")

# also a pure-LGBM grid as a sanity baseline (smooth, well-calibrated signal)
print("Writing candidate submissions (LightGBM only):")
save_grid(lgbm_probs, "lgbm")

# pure cross-encoder grid (this CE alone already scored 0.72 at ~38%)
print("Writing candidate submissions (cross-encoder only):")
save_grid(ce_probs, "ce")

# default submission.csv = blend @ 40% (middle of the grid)
k = int(round(0.40 * len(combined)))
order = np.argsort(-combined)
preds = np.zeros(len(combined), dtype=np.int8); preds[order[:k]] = 1
pd.DataFrame({"id": sub["id"].values, "prediction": preds}).to_csv(HERE / f"submission{OUT_SUFFIX}.csv", index=False)
print(f"\nDefault submission{OUT_SUFFIX}.csv = blend @ 40%  ({preds.sum():,} positive)")
print("Submit the p30/p35/p40/p45/p50 variants and keep whichever scores best.")
