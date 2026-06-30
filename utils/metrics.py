"""Метрики качества моделей без тяжелых внешних зависимостей."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_binary_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"y": y_true, "score": y_score}).replace([np.inf, -np.inf], np.nan).dropna()
    y = frame["y"].astype(int).to_numpy()
    score = frame["score"].astype(float).to_numpy()
    return y, score


def roc_auc_score(y_true, y_score) -> float:
    """Посчитать ROC AUC через rank statistic с обработкой ties."""
    y, score = _clean_binary_arrays(y_true, y_score)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(score).rank(method="average").to_numpy()
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_score(y_true, y_score) -> float:
    """Посчитать Average Precision для бинарной классификации."""
    y, score = _clean_binary_arrays(y_true, y_score)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1.0)
    return float((precision * y_sorted).sum() / n_pos)


def brier_score(y_true, y_score) -> float:
    y, score = _clean_binary_arrays(y_true, y_score)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((y - score) ** 2))


def _pearson_corr(x, y) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2:
        return float("nan")
    x_arr = frame["x"].astype(float).to_numpy()
    y_arr = frame["y"].astype(float).to_numpy()
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def evaluate_model_quality(scored: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Оценить бинарные головы на фактически наблюдаемых исходах."""
    frame = scored.copy()
    if split is not None:
        frame = frame.loc[frame["split"] == split].copy()

    specs = [
        ("response_model", "response", "shown", "p_util_hat"),
        ("response_transformer", "response", "shown", "transformer_p_util_hat"),
        ("response_stacked", "response", "shown", "stacked_p_util_hat"),
        ("control_model", "control", "not_shown", "p0_hat"),
        ("control_transformer", "control", "not_shown", "transformer_p0_hat"),
        ("control_stacked", "control", "not_shown", "stacked_p0_hat"),
    ]

    rows = []
    for model_name, head, subset, score_col in specs:
        if score_col not in frame.columns:
            continue
        if subset == "shown":
            model_frame = frame.loc[frame["shown"]].copy()
        elif subset == "not_shown":
            model_frame = frame.loc[~frame["shown"]].copy()
        else:
            model_frame = frame.copy()

        y = model_frame["utilized"].astype(int)
        score = model_frame[score_col].astype(float)
        rows.append(
            {
                "model_name": model_name,
                "head": head,
                "split": split or "all",
                "subset": subset,
                "n_rows": int(len(model_frame)),
                "positive_rate": float(y.mean()) if len(y) else float("nan"),
                "mean_score": float(score.mean()) if len(score) else float("nan"),
                "roc_auc": roc_auc_score(y, score),
                "average_precision": average_precision_score(y, score),
                "brier": brier_score(y, score),
            }
        )

    return pd.DataFrame(rows)


def evaluate_treatment_overlap(
    assignments: pd.DataFrame,
    train_split: str = "train",
    eval_splits: tuple[str, ...] = ("val", "test"),
    seed: int = 42,
) -> pd.DataFrame:
    """Причинная overlap-диагностика рандомизированной раздачи.

    Обучает простую модель предсказывать факт показа `shown` по предраздаточным
    признакам и оценивает её ROC AUC на отложенных срезах. Значение около 0.5
    означает, что показ не предсказуем по признакам до раздачи, то есть в логах
    нет скрытой селекции и причинная интерпретация uplift надёжна. Заметное
    превышение 0.5 сигнализировало бы о нарушении рандомизации.
    """
    from utils.models import (
        SimpleGradientBoostingClassifier,
        infer_feature_columns,
        make_feature_matrix,
    )

    feature_columns = [col for col in infer_feature_columns(assignments) if col != "is_available"]
    train = assignments.loc[(assignments["split"] == train_split) & assignments["is_available"]].copy()
    if train.empty or train["shown"].nunique() < 2:
        return pd.DataFrame(columns=["split", "n_rows", "shown_rate", "roc_auc_shown"])

    x_train, training_columns = make_feature_matrix(train, feature_columns)
    y_train = train["shown"].astype(int).to_numpy()
    model = SimpleGradientBoostingClassifier(n_estimators=80, random_state=seed)
    model.fit(x_train.to_numpy(), y_train)

    rows = []
    for split in eval_splits:
        frame = assignments.loc[(assignments["split"] == split) & assignments["is_available"]].copy()
        if frame.empty or frame["shown"].nunique() < 2:
            continue
        x_eval, _ = make_feature_matrix(frame, feature_columns, training_columns=training_columns)
        p_shown = model.predict_proba(x_eval.to_numpy())[:, 1]
        rows.append(
            {
                "split": split,
                "n_rows": int(len(frame)),
                "shown_rate": float(frame["shown"].mean()),
                "roc_auc_shown": roc_auc_score(frame["shown"].astype(int), p_shown),
            }
        )
    return pd.DataFrame(rows)


def evaluate_uplift_alignment(scored: pd.DataFrame, split: str | None = None, top_share: float = 0.10) -> pd.DataFrame:
    """Сравнить предсказанный uplift с синтетическим истинным эффектом."""
    frame = scored.copy()
    if split is not None:
        frame = frame.loc[frame["split"] == split].copy()
    frame = frame.loc[frame["is_available"]].copy()

    specs = [
        ("extra_npv_model", "uplift_hat"),
        ("extra_npv_transformer", "transformer_uplift_hat"),
        ("extra_npv_stacked", "stacked_uplift_hat"),
    ]

    rows = []
    for model_name, uplift_col in specs:
        if uplift_col not in frame.columns:
            continue

        pred_extra_npv = frame["npv"].astype(float) * frame[uplift_col].astype(float)
        ranked = frame.assign(pred_extra_npv=pred_extra_npv).sort_values(
            "pred_extra_npv",
            ascending=False,
            kind="mergesort",
        )
        top_n = max(1, int(np.ceil(len(ranked) * top_share)))
        top_frame = ranked.head(top_n)
        overall_true_extra = float(ranked["extra_npv_true"].mean())
        top_true_extra = float(top_frame["extra_npv_true"].mean())

        rows.append(
            {
                "model_name": model_name,
                "split": split or "all",
                "n_rows": int(len(ranked)),
                "delta_corr": _pearson_corr(ranked["delta_p_true"], ranked[uplift_col]),
                "extra_npv_corr": _pearson_corr(ranked["extra_npv_true"], ranked["pred_extra_npv"]),
                "overall_true_extra_npv": overall_true_extra,
                "top_decile_true_extra_npv": top_true_extra,
                "top_decile_lift": top_true_extra / overall_true_extra if overall_true_extra else float("nan"),
            }
        )

    return pd.DataFrame(rows)
