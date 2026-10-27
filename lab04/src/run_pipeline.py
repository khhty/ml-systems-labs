import hashlib
import json
import logging
import platform
import sys
from pathlib import Path

import joblib
import numpy
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lab4_cleaner import Cleaner

BASE = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE / "config.yaml", encoding="utf-8"))
SEED = CFG["seed"]
RES = BASE / CFG["paths"]["results"]
RES.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(RES / "pipeline.log", mode="w", encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger()

TYPES = CFG["schema"]["types"]
NUM = [c for c, t in TYPES.items() if t == "int"]
CAT = [c for c, t in TYPES.items() if t == "category" and c != "income"]
DERIVED = ["capital_net", "hours_per_age"]

def ingest():
    df = fetch_openml(name="adult", version=2, as_frame=True).frame
    df = df.rename(columns={"class": "income"})
    log.info("INGEST rows=%d cols=%d", *df.shape)
    exp = (CFG["expected"]["rows"], CFG["expected"]["cols"])
    assert df.shape == exp, f"Expected {exp}, got {df.shape}"
    return df

def validate(df):
    rep = {"rows": len(df), "cols": df.shape[1], "type_violations": [],
           "range_violations": {}, "missing": {}}
    for col, t in TYPES.items():
        if col not in df.columns:
            rep["type_violations"].append(f"{col}: column missing")
            continue
        s = df[col]
        if t == "int":
            ok = pd.api.types.is_numeric_dtype(s) and bool((s.dropna() % 1 == 0).all())
        else:
            ok = isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object
        if not ok:
            rep["type_violations"].append(f"{col}: expected {t}, got {s.dtype}")
        miss = s.isna() | (s.astype(object) == "?")
        rep["missing"][col] = int(miss.sum())
    for col, (lo, hi) in CFG["schema"]["ranges"].items():
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        rep["range_violations"][col] = bad
    rep["exact_duplicate_rows"] = int(df.duplicated().sum())
    rep["income_values_ok"] = set(df["income"].dropna().unique()) <= {"<=50K", ">50K"}
    rep["categories"] = {c: sorted(map(str, df[c].dropna().unique()))
                         for c, t in TYPES.items() if t == "category"}
    json.dump(rep, open(RES / "validation_report.json", "w"), indent=2)
    log.info("VALIDATION %s", json.dumps(rep))
    return rep

def add_features(X):
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16).clip(lower=1)
    return X

def evaluate(Xtr, ytr, Xte, yte, num_cols):
    pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return {"accuracy": float(accuracy_score(yte, p)), "f1": float(f1_score(yte, p))}

def label_audit(df):
    feats = [c for c in df.columns if c != "income"]
    g = df.groupby(feats, dropna=False, observed=True)["income"].agg(["nunique", "size"])
    conflict = g[g["nunique"] > 1]
    out = {
        "conflicting_feature_groups": int(len(conflict)),
        "rows_in_conflicting_groups": int(conflict["size"].sum()),
        "education_vs_education_num_consistent":
            bool(df.groupby("education", observed=True)["education-num"].nunique().max() == 1),
    }
    sample = df.sample(50, random_state=SEED)
    sample.to_csv(RES / "sample_50_rows.csv", index=True)
    out["sample50_range_violations"] = int(sum(
        ((sample[c] < lo) | (sample[c] > hi)).sum()
        for c, (lo, hi) in CFG["schema"]["ranges"].items()))
    json.dump(out, open(RES / "label_audit.json", "w"), indent=2)
    log.info("LABEL AUDIT %s", out)
    return out

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_versions():
    lines = [f"Python {sys.version.split()[0]}", f"Platform {platform.platform()}",
             f"numpy=={numpy.__version__}", f"pandas=={pd.__version__}",
             f"scikit-learn=={sklearn.__version__}", f"joblib=={joblib.__version__}",
             f"pyyaml=={yaml.__version__}"]
    (RES / "versions.txt").write_text("\n".join(lines) + "\n")

def main():
    write_versions()
    raw = ingest()
    validate(raw)
    label_audit(raw)

    X = raw.drop(columns="income")
    y = (raw["income"] == ">50K").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=CFG["test_size"], stratify=y, random_state=SEED)
    log.info("SPLIT train=%d test=%d", len(Xtr), len(Xte))

    trn = Xtr.assign(income=ytr)
    n0 = len(trn)
    trn = trn.drop_duplicates()
    log.info("DUPLICATES dropped from train: %d", n0 - len(trn))
    Xtr, ytr = trn.drop(columns="income"), trn["income"]

    lo, hi = CFG["clip_percentiles"]
    cleaner = Cleaner(NUM, CAT, lo, hi).fit(Xtr)  
    joblib.dump(cleaner, BASE / CFG["paths"]["preprocessor"])
    Xtr_c, Xte_c = cleaner.transform(Xtr), cleaner.transform(Xte)

    m_base = evaluate(Xtr_c, ytr, Xte_c, yte, NUM)
    Xtr_f, Xte_f = add_features(Xtr_c), add_features(Xte_c)
    m_feat = evaluate(Xtr_f, ytr, Xte_f, yte, NUM + DERIVED)
    log.info("BASELINE %s", m_base)
    log.info("WITH FEATURES %s", m_feat)

    proc = pd.concat([Xtr_f.assign(income=ytr, split="train"),
                      Xte_f.assign(income=yte, split="test")])
    path = BASE / CFG["paths"]["processed"]
    proc.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    digest = sha256_file(path)
    (RES / "data_hash.txt").write_text(digest + "\n")
    log.info("SHA256 %s", digest)

    summary = {"config": CFG, "sha256": digest, "processed_shape": list(proc.shape),
               "metrics_baseline": m_base, "metrics_with_features": m_feat}
    yaml.safe_dump(summary, open(RES / "run_summary.yaml", "w"), sort_keys=False)

if __name__ == "__main__":
    main()