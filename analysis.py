"""Healthy vs pathological gait: criticality-only logistic regression vs criticality+gait-stats tree ensemble."""
import json, os, sys, time
import numpy as np, pandas as pd, requests
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from lightgbm import LGBMClassifier

SEED, SEEDS = 42, [42, 43, 44]
BASE = "https://physionet.org/files/gaitndd/1.0.0/"
np.random.seed(SEED)

def get(url):  # download with timeout and retry/backoff on 429/5xx
    for k in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200: return r.text
            if r.status_code not in (429, 500, 502, 503, 504): return None
        except requests.RequestException: pass
        time.sleep(2 ** k)
    return None

def record_names():
    txt = get(BASE + "RECORDS")
    names = [l.strip().replace(".ts", "") for l in (txt or "").splitlines() if l.strip()]
    names = [n for n in names if n.startswith(("control", "als", "hunt", "park"))]
    if len(names) < 50:  # fallback to the documented naming scheme
        names = [f"{p}{i}" for p, n in [("control", 16), ("als", 13), ("hunt", 20), ("park", 15)] for i in range(1, n + 1)]
    return names

# ---- criticality-style features ----
def loglog_slope(scales, vals): return np.polyfit(np.log(scales), np.log(vals), 1)[0]

def dfa(x):  # detrended fluctuation analysis exponent (linear detrending)
    y = np.cumsum(x - x.mean())
    sc = np.unique(np.logspace(np.log10(4), np.log10(len(x) // 4), 12).astype(int))
    F = []
    for s in sc:
        seg = y[: len(y) // s * s].reshape(-1, s); t = np.arange(s)
        c = np.polyfit(t, seg.T, 1)
        F.append(np.sqrt(np.mean((seg - (c[0][:, None] * t + c[1][:, None])) ** 2)))
    return loglog_slope(sc, F)

def hurst(x):  # rescaled-range exponent
    sc = np.unique(np.logspace(np.log10(8), np.log10(len(x) // 2), 10).astype(int)); rs = []
    for s in sc:
        seg = x[: len(x) // s * s].reshape(-1, s)
        dev = np.cumsum(seg - seg.mean(1, keepdims=True), 1)
        rs.append(np.mean((dev.max(1) - dev.min(1)) / (seg.std(1) + 1e-12)))
    return loglog_slope(sc, rs)

def sampen(x, m=2, r=0.2):
    x = (x - x.mean()) / x.std(); n = len(x)
    def cnt(mm):
        t = sliding_window_view(x, mm)[: n - m]
        d = np.abs(t[:, None, :] - t[None, :, :]).max(-1)
        return np.sum(np.triu(d < r, 1))
    a, b = cnt(m + 1), cnt(m)
    return -np.log(a / b) if a > 0 and b > 0 else np.nan

def tail_exponent(x):  # MLE power-law exponent of large-fluctuation sizes (top 25%)
    z = np.abs(x - np.median(x)) / x.std(); xmin = np.percentile(z, 75)
    tail = z[z >= xmin]; s = np.sum(np.log(tail / xmin))
    return 1 + len(tail) / s if xmin > 0 and s > 0 else np.nan

CRIT = ["dfa", "hurst", "sampen", "acf1", "tail_alpha"]

def features(df):
    a = df.values.astype(float)
    L, R = a[:, 1], a[:, 2]; mL, mR = np.median(L), np.median(R)
    ok = (L > .5 * mL) & (L < 1.5 * mL) & (R > .5 * mR) & (R < 1.5 * mR) & (a[:, 3] > 0) & (a[:, 4] > 0) & (a[:, 11] > 0)
    a = a[ok]; L, R = a[:, 1], a[:, 2]; x = (L + R) / 2
    f = {"dfa": dfa(x), "hurst": hurst(x), "sampen": sampen(x), "acf1": np.corrcoef(x[:-1], x[1:])[0, 1],
         "tail_alpha": tail_exponent(x)}
    f.update(mean_stride=x.mean(), cv=x.std() / x.mean(),
             swing_ratio=np.mean(a[:, 3] / L), stance_ratio=np.mean(a[:, 7] / L), ds_ratio=np.mean(a[:, 11] / x),
             stride_asym=abs(L.mean() - R.mean()) / x.mean(),
             swing_asym=abs(a[:, 3].mean() - a[:, 4].mean()) / a[:, 3:5].mean())
    return f

def load():
    os.makedirs("data", exist_ok=True); rows = []
    for n in record_names():
        p = f"data/{n}.ts"
        if not os.path.exists(p):
            t = get(BASE + n + ".ts")
            if t is None: continue
            open(p, "w").write(t)
        try:
            df = pd.read_csv(p, sep=r"\s+", header=None)
            if df.shape[1] < 13 or len(df) < 60: continue
            rows.append({"name": n, "y": int(not n.startswith("control")), **features(df)})
        except Exception as e: print("skip", n, e)
    return pd.DataFrame(rows)

# ---- models (shared inner CV + preprocessing; only features/estimator differ) ----
def fit_baseline(X, y, seed):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
    grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=seed)).fit(X, y)

def fit_improved(X, y, seed):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", "passthrough")])
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1)
    lgb = LGBMClassifier(n_estimators=100, min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1)
    grid = [{"clf": [rf], "clf__max_depth": [3, None], "clf__min_samples_leaf": [1, 3]},
            {"clf": [lgb], "clf__learning_rate": [0.05, 0.1], "clf__num_leaves": [4, 8]}]
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=seed)).fit(X, y)

def evaluate(m, X, y):  # shared test metrics
    p = m.predict_proba(X)[:, 1]; yh = (p >= 0.5).astype(int)
    return [roc_auc_score(y, p), balanced_accuracy_score(y, yh), f1_score(y, yh)]

def run_split(d, tr, te, seeds):
    y = d["y"].values; out = {"b": [], "i": []}; choice = []
    for s in seeds:
        mb = fit_baseline(d.loc[tr, CRIT], y[tr], s); out["b"].append(evaluate(mb, d.loc[te, CRIT], y[te]))
        mi = fit_improved(d.loc[tr, ALL], y[tr], s); out["i"].append(evaluate(mi, d.loc[te, ALL], y[te]))
        choice.append(type(mi.best_estimator_.named_steps["clf"]).__name__)
    return {k: np.array(v) for k, v in out.items()}, choice

d = load()
print("recordings loaded:", len(d), "| pathological:", int(d.y.sum()))
if len(d) < 50 or d.y.nunique() < 2: sys.exit("Not enough data")
ALL = [c for c in d.columns if c not in ("name", "y")]
idx = np.arange(len(d)); y = d["y"].values

# Primary: one stratified 70/30 subject-level split, 3 model seeds
tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)
res, choice = run_split(d, tr, te, SEEDS)
print("primary split done; improved chose:", choice)

# Supplementary: 20 repeated stratified 70/30 splits (model seed 42)
reps = {"b": [], "i": []}
for k, (a, b) in enumerate(StratifiedShuffleSplit(20, test_size=0.3, random_state=SEED).split(idx, y)):
    r, _ = run_split(d, a, b, [SEED]); reps["b"].append(r["b"][0]); reps["i"].append(r["i"][0])
    if k % 5 == 4: print(f"repeated splits {k + 1}/20")
reps = {k: np.array(v) for k, v in reps.items()}

M = {}
for tag, key in [("baseline", "b"), ("improved", "i")]:
    m, s = res[key].mean(0), res[key].std(0)
    M[f"{tag}_score"], M[f"{tag}_score_std"] = m[0], s[0]
    M[f"{tag}_balanced_accuracy"], M[f"{tag}_f1"] = m[1], m[2]
    rm, rs = reps[key].mean(0), reps[key].std(0)
    M[f"{tag}_repeated_auc_mean"], M[f"{tag}_repeated_auc_std"] = rm[0], rs[0]
    M[f"{tag}_repeated_balacc_mean"], M[f"{tag}_repeated_f1_mean"] = rm[1], rm[2]
M["improvement"] = M["improved_score"] - M["baseline_score"]
M["improvement_repeated_auc"] = M["improved_repeated_auc_mean"] - M["baseline_repeated_auc_mean"]
M["repeated_fraction_splits_improved_better"] = float(np.mean(reps["i"][:, 0] > reps["b"][:, 0]))
M = {k: float(v) for k, v in M.items()}
if not all(np.isfinite(list(M.values()))): sys.exit("Non-finite metrics")
print(json.dumps(M, indent=2))

os.makedirs("figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4)); w = 0.35
for j, (nm, key) in enumerate([("baseline", "b"), ("improved", "i")]):
    ax.bar(np.arange(3) + j * w, reps[key].mean(0), w, yerr=reps[key].std(0), label=nm, capsize=3)
ax.set_xticks(np.arange(3) + w / 2); ax.set_xticklabels(["ROC-AUC", "Bal. acc.", "F1"]); ax.legend()
ax.set_title("Test metrics, 20 repeated splits"); fig.tight_layout(); fig.savefig("figures/comparison.png", dpi=120)

out = {
    "question": "Can a tuned gradient-boosting classifier on criticality plus standard gait-variability features beat a criticality-only classifier at separating healthy from pathological gait in stride-interval recordings?",
    "method": "Per-recording stride series -> criticality features (DFA, Hurst, sample entropy, lag-1 ACF, power-law tail exponent) and gait statistics. Headline: one stratified 70/30 subject-level split (seed 42), models run with seeds 42-44; supplementary: 20 repeated stratified 70/30 splits. Both models tuned by 3-fold inner CV on the training split only; test set used once.",
    "dataset": {"name": "PhysioNet Gait in Neurodegenerative Disease (gaitndd)", "source_url": "https://physionet.org/content/gaitndd/1.0.0/",
                "n_rows": int(len(d)), "n_train": int(len(tr)), "n_test": int(len(te)), "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "ROC-AUC", "higher_is_better": True,
    "baseline": "Logistic regression (C tuned by inner CV) on the five criticality-style features only (stand-in for the paper's undisclosed criticality method).",
    "improved": "Criticality features plus gait statistics (mean, CV, swing/stance/double-support ratios, L-R asymmetry), with a random forest or LightGBM chosen/tuned by inner CV.",
    "metrics": M,
    "notes": f"Stand-in dataset (neurodegenerative gait, not metabolic disorders); pathological = ALS/Huntington/Parkinson pooled. Only {len(d)} subjects, so test sets are ~{len(te)} subjects and scores are noisy; the repeated-split metrics are supplementary and use model seed 42 only. Baseline LR is deterministic so its seed std reflects only inner-CV fold shuffling. Model families chosen on primary split by seed: {choice}. Criticality features are simple estimators computed on the mean of left/right stride intervals after crude outlier removal.",
}
json.dump(out, open("results.json", "w", encoding="utf-8"), indent=2)
print("wrote results.json")
