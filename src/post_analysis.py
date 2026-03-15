#!/usr/bin/env python3
"""Post-hoc error analysis for Lying Style experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    df = pd.read_csv(RESULTS / "all_outputs_with_features.csv")
    work = df[(df["seed"] == 42) & (df["condition"].isin(["truthful", "lie_roleplay"]))].copy()
    work["label"] = (work["condition"] != "truthful").astype(int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(splitter.split(work, y=work["label"], groups=work["question_id"]))
    train_df = work.iloc[train_idx].copy()
    test_df = work.iloc[test_idx].copy()

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    x_train = vec.fit_transform(train_df["text"])
    x_test = vec.transform(test_df["text"])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train, train_df["label"])
    proba = clf.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(test_df["label"], pred)
    cm_payload = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    with (RESULTS / "confusion_matrix.json").open("w", encoding="utf-8") as f:
        json.dump(cm_payload, f, indent=2)

    error_df = test_df.copy()
    error_df["pred"] = pred
    error_df["proba_lie"] = proba
    error_df["error_type"] = np.where(
        (error_df["label"] == 0) & (error_df["pred"] == 1),
        "false_positive",
        np.where((error_df["label"] == 1) & (error_df["pred"] == 0), "false_negative", "correct"),
    )

    fp = error_df[error_df["error_type"] == "false_positive"].sort_values("proba_lie", ascending=False)
    fn = error_df[error_df["error_type"] == "false_negative"].sort_values("proba_lie", ascending=True)
    fp.head(20).to_csv(RESULTS / "error_false_positives.csv", index=False)
    fn.head(20).to_csv(RESULTS / "error_false_negatives.csv", index=False)


if __name__ == "__main__":
    main()
