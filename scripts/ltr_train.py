"""Train and judge the small LTR combiner under the pre-registered lines.

Regimes: A→A and B→B as GroupKFold(5)-by-query cross-validation (honest
within-anchor), plus the transfer cells A→B and B→A (train all of one
anchor, predict all of the other) — the cells that decide whether the
deterministic feature space generalises across relevance definitions.

Metrics per target query: any positive session in the learned top-2 / top-8
versus the shipped order (the ``rank`` feature). Queries with no positive
in the deep-24 are excluded from metrics (the ranker cannot fix recall)
and their count is reported.

Gate (pre-registration §3): a transfer cell must beat the shipped baseline
on the target anchor's top-2 membership for the e2e arm to be built.

Usage::

    PYTHONPATH=src python scripts/ltr_train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "base", "coverage", "strength", "entity", "f1", "rare_cov", "f3",
    "rank", "final", "n_members",
]


def load(anchor: str) -> list[dict]:
    data = json.loads(
        (ROOT / "eval/results/ltr_data.json").read_text(encoding="utf-8")
    )
    return [r for r in data["rows"] if r["anchor"] == anchor]


def matrices(rows: list[dict]):
    X, y, groups = [], [], []
    for i, r in enumerate(rows):
        for s in r["sessions"]:
            X.append([s["features"][f] for f in FEATURES])
            y.append(s["label"])
            groups.append(i)
    return np.asarray(X, dtype=float), np.asarray(y), np.asarray(groups)


def standardize(X_train: np.ndarray, X_apply: np.ndarray):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_apply - mu) / sd, mu, sd


def predict_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def hit_metrics(rows: list[dict], scores_per_query: list[dict[str, float]],
                k: int) -> tuple[int, int]:
    """Queries whose top-k learned order holds any positive."""
    hit = total = 0
    for r, scores in zip(rows, scores_per_query):
        if not any(s["label"] for s in r["sessions"]):
            continue
        total += 1
        top = sorted(scores, key=lambda sid: -scores[sid])[:k]
        if any(s["label"] for s in r["sessions"] if s["sid"] in top):
            hit += 1
    return hit, total


def baseline_hits(rows: list[dict], k: int) -> tuple[int, int]:
    hit = total = 0
    for r in rows:
        if not any(s["label"] for s in r["sessions"]):
            continue
        total += 1
        top = {s["sid"] for s in r["sessions"][:k]}
        if any(s["label"] for s in r["sessions"] if s["sid"] in top):
            hit += 1
    return hit, total


def paired(rows: list[dict], scores_per_query: list[dict[str, float]],
           k: int) -> tuple[int, int]:
    """Per-query win/loss of learned top-k vs shipped top-k."""
    win = loss = 0
    for r, scores in zip(rows, scores_per_query):
        if not any(s["label"] for s in r["sessions"]):
            continue
        labels = {s["sid"]: s["label"] for s in r["sessions"]}
        top = set(sorted(scores, key=lambda sid: -scores[sid])[:k])
        base = {s["sid"] for s in r["sessions"][:k]}
        l_hit = any(labels[s] for s in top)
        b_hit = any(labels[s] for s in base)
        if l_hit and not b_hit:
            win += 1
        if b_hit and not l_hit:
            loss += 1
    return win, loss


def flat_cv_scores(rows: list[dict], make_model) -> list[dict[str, float]]:
    """GroupKFold by query; every (query, session) of the held-out queries
    is scored by the fold's model."""
    X, y, groups = matrices(rows)
    by_query: dict[int, dict[str, float]] = {i: {} for i in range(len(rows))}
    session_ids = [
        [s["sid"] for s in r["sessions"]] for r in rows
    ]
    gkf = GroupKFold(n_splits=5)
    for tr_idx, te_idx in gkf.split(X, y, groups):
        Xtr, Xte = X[tr_idx], X[te_idx]
        Xtr_s, Xte_s, _, _ = standardize(Xtr, Xte)
        if len(np.unique(y[tr_idx])) < 2:
            continue
        model = make_model()
        model.fit(Xtr_s, y[tr_idx])
        scores = predict_scores(model, Xte_s)
        for pos, sc in zip(te_idx, scores):
            qi = int(groups[pos])
            prior = sum(len(session_ids[g]) for g in range(qi))
            within = pos - prior
            by_query[qi][session_ids[qi][within]] = float(sc)
    return [by_query[i] for i in range(len(rows))]


def transfer_scores(train_rows: list[dict], test_rows: list[dict],
                    make_model) -> tuple[list[dict[str, float]], object]:
    Xtr, ytr, _ = matrices(train_rows)
    Xte, _, _ = matrices(test_rows)
    Xtr_s, Xte_s, mu, sd = standardize(Xtr, Xte)
    model = make_model()
    if len(np.unique(ytr)) < 2:
        raise SystemExit("training side has a single class")
    model.fit(Xtr_s, ytr)
    scores = predict_scores(model, Xte_s)
    out: list[dict[str, float]] = [dict() for _ in test_rows]
    pos = 0
    for i, r in enumerate(test_rows):
        for s in r["sessions"]:
            out[i][s["sid"]] = float(scores[pos])
            pos += 1
    return out, (model, mu, sd)


def report(name: str, rows: list[dict],
           scores: list[dict[str, float]]) -> dict:
    line = {"regime": name}
    for k in (2, 8):
        hit, total = hit_metrics(rows, scores, k)
        bhit, _ = baseline_hits(rows, k)
        win, loss = paired(rows, scores, k)
        line[f"top{k}_hit"] = hit
        line[f"top{k}_baseline"] = bhit
        line[f"top{k}_total"] = total
        line[f"top{k}_winloss"] = f"+{win}/-{loss}"
        print(f"  {name:8s} top-{k}: learned {hit}/{total}  "
              f"shipped {bhit}/{total}  paired +{win}/-{loss}")
    return line


def main() -> int:
    A, B = load("A"), load("B")
    print(f"anchor A: {len(A)} queries; anchor B: {len(B)} queries")

    results = []

    def logistic():
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")

    def gbdt():
        return GradientBoostingClassifier(
            max_depth=3, n_estimators=50, learning_rate=0.1
        )

    for model_name, make in (("logistic", logistic), ("gbdt", gbdt)):
        print(f"\n{model_name}:")
        results.append(report(f"A→A(cv)", A, flat_cv_scores(A, make)))
        results.append(report(f"B→B(cv)", B, flat_cv_scores(B, make)))
        ab, ab_model = transfer_scores(A, B, make)
        results.append(report("A→B", B, ab))
        ba, _ = transfer_scores(B, A, make)
        results.append(report("B→A", A, ba))
        if model_name == "logistic":
            model, mu, sd = ab_model
            weights = {
                f: round(float(w), 4) for f, w in zip(FEATURES, model.coef_[0])
            }
            print(f"  A→B logistic weights: {weights}")
            Path(ROOT / "eval/results/ltr_logistic_AB.json").write_text(
                json.dumps({
                    "features": FEATURES, "weights": weights,
                    "mu": [round(float(m), 5) for m in mu],
                    "sd": [round(float(s), 5) for s in sd],
                }, indent=1), encoding="utf-8"
            )

    # gate: best transfer cell vs shipped on target top-2
    gate_pass = False
    for r in results:
        if r["regime"] in ("A→B", "B→A") and r["top2_hit"] > r["top2_baseline"]:
            gate_pass = True
    print(f"\noffline gate (transfer beats shipped top-2): "
          f"{'PASS' if gate_pass else 'FAIL — no e2e arm'}")

    with (ROOT / "eval/results/ltr_train.json").open("w", encoding="utf-8") as h:
        json.dump({"results": results, "gate_pass": gate_pass}, h, indent=1)
    print("wrote eval/results/ltr_train.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
