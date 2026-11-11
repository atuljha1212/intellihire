"""ML Feature Scoring & Classification Pipeline (Scikit-learn + XGBoost).

The weighted equation ranks candidates from day one. As recruiters mark
candidates shortlisted/interviewed (positive) or rejected (negative), this
module learns from those decisions: it cross-validates Logistic Regression,
Random Forest and XGBoost on the five sub-scores, keeps the best by ROC-AUC,
and exposes its probability as an `ml_score` plus feature importances - so
the recruiter can see which signals actually drive their own decisions.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from intellihire.scoring.features import ML_FEATURE_NAMES

log = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - depends on environment
    XGBClassifier = None
    HAS_XGBOOST = False

MIN_LABELS = 6
MIN_PER_CLASS = 2
POSITIVE_STATUSES = {"shortlisted", "interview", "hired"}
NEGATIVE_STATUSES = {"rejected"}


class NotEnoughLabels(Exception):
    pass


@dataclass
class TrainingReport:
    best_model: str
    cv_auc: dict[str, float]
    cv_folds: int
    n_samples: int
    n_positive: int
    feature_importance: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def status_to_label(status: str) -> int | None:
    if status in POSITIVE_STATUSES:
        return 1
    if status in NEGATIVE_STATUSES:
        return 0
    return None


def _candidates(n_samples: int, pos_weight: float = 1.0) -> dict[str, object]:
    models: dict[str, object] = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=2, class_weight="balanced", random_state=42
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
            min_child_weight=1, eval_metric="logloss", scale_pos_weight=pos_weight, random_state=42, n_jobs=1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier

        models["Gradient Boosting (XGBoost not installed)"] = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, random_state=42
        )
    return models


def _importance(model, name: str) -> np.ndarray:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    else:  # pragma: no cover
        imp = np.ones(len(ML_FEATURE_NAMES))
    total = imp.sum()
    return imp / total if total > 0 else imp


def train_ranker(X: np.ndarray, y: np.ndarray, save_path: Path | None = None) -> tuple[object, TrainingReport]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if len(y) < MIN_LABELS or min(n_pos, n_neg) < MIN_PER_CLASS:
        raise NotEnoughLabels(
            f"Need at least {MIN_LABELS} labelled candidates with {MIN_PER_CLASS}+ in each class "
            f"(have {n_pos} positive, {n_neg} negative). Shortlist or reject more candidates first."
        )
    folds = max(2, min(5, n_pos, n_neg))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs: dict[str, float] = {}
    models = _candidates(len(y), n_neg / max(n_pos, 1))
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", error_score="raise")
            aucs[name] = round(float(np.mean(scores)), 4)  # a NaN fold fails the model instead of being averaged away
        except Exception as exc:  # e.g. a fold with a single class
            log.warning("CV failed for %s: %s", name, exc)
            aucs[name] = float("nan")
    valid = {k: v for k, v in aucs.items() if not np.isnan(v)}
    best_name = max(valid, key=valid.get) if valid else "Logistic Regression"
    best = models[best_name]
    best.fit(X, y)

    notes = []
    if len(y) < 30:
        notes.append("Small training set - treat the learned ranking as indicative only.")
    report = TrainingReport(
        best_model=best_name,
        cv_auc=aucs,
        cv_folds=folds,
        n_samples=int(len(y)),
        n_positive=n_pos,
        feature_importance={f: round(float(v), 4) for f, v in zip(ML_FEATURE_NAMES, _importance(best, best_name))},
        notes=notes,
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": best, "features": ML_FEATURE_NAMES, "report": report.to_dict()}, save_path)
    return best, report


def load_ranker(path: Path):
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        return bundle["model"]
    except Exception as exc:  # pragma: no cover
        log.warning("Could not load ranker %s: %s", path, exc)
        return None


def predict_scores(model, X: np.ndarray) -> np.ndarray:
    if model is None or len(X) == 0:
        return np.zeros(len(X))
    return model.predict_proba(np.asarray(X, dtype=float))[:, 1]
