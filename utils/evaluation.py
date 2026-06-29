"""Оффлайн-оценка стратегий: IPS, SNIPS, ESS и бутстреп."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


UNIT_COLUMNS = ["user_id", "period_id"]
MERGE_COLUMNS = ["user_id", "period_id", "category_id"]

PREDICTED_P1_COLUMNS = [
    "p1_hat",
    "transformer_p1_hat",
    "stacked_p1_hat",
]

DR_Q_COLUMNS_BY_POLICY = {
    "response_model": ["p1_hat"],
    "extra_npv_model": ["p1_hat"],
    "response_transformer": ["transformer_p1_hat", "p1_hat"],
    "extra_npv_transformer": ["transformer_p1_hat", "p1_hat"],
    "response_stacked": ["stacked_p1_hat", "p1_hat"],
    "extra_npv_stacked": ["stacked_p1_hat", "p1_hat"],
    "random": ["p1_hat"],
    "npv": ["p1_hat"],
    "response_true": ["p1_true"],
    "oracle_extra_npv": ["p1_true"],
}


def make_policy_eval_frame(
    assignments: pd.DataFrame,
    policy_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Соединить выбранные стратегией категории с randomized логом."""
    required_assignments = {
        "user_id",
        "period_id",
        "split",
        "category_id",
        "group_id",
        "shown",
        "show_propensity",
        "utilized",
        "npv",
        "p1_true",
        "extra_npv_true",
    }
    required_policy = {
        "user_id",
        "period_id",
        "category_id",
        "policy_name",
        "rank",
        "policy_score",
    }
    missing_assignments = required_assignments - set(assignments.columns)
    missing_policy = required_policy - set(policy_assignments.columns)
    if missing_assignments:
        raise ValueError(f"В assignments не хватает колонок: {sorted(missing_assignments)}")
    if missing_policy:
        raise ValueError(f"В policy_assignments не хватает колонок: {sorted(missing_policy)}")

    factual_columns = [
        "user_id",
        "period_id",
        "split",
        "category_id",
        "group_id",
        "shown",
        "show_propensity",
        "utilized",
        "npv",
        "p1_true",
        "extra_npv_true",
    ]
    factual_columns.extend([col for col in PREDICTED_P1_COLUMNS if col in assignments.columns])
    factual = assignments[factual_columns].copy()
    policy = policy_assignments[
        ["user_id", "period_id", "category_id", "policy_name", "rank", "policy_score"]
    ].copy()

    frame = policy.merge(factual, on=MERGE_COLUMNS, how="left", validate="many_to_one")
    if frame["shown"].isna().any():
        raise ValueError("Не все выбранные политикой категории найдены в assignments.")

    frame["matched"] = frame["shown"].astype(bool)
    frame["weight"] = np.where(frame["matched"], 1.0 / frame["show_propensity"].clip(lower=1e-9), 0.0)
    frame["observed_value"] = frame["utilized"].astype(float) * frame["npv"].astype(float)
    frame["weighted_value"] = frame["weight"] * frame["observed_value"]
    frame["true_shown_value"] = frame["p1_true"].astype(float) * frame["npv"].astype(float)
    frame["true_extra_npv_value"] = frame["extra_npv_true"].astype(float)
    frame["q_value"] = np.nan
    for policy_name, candidate_cols in DR_Q_COLUMNS_BY_POLICY.items():
        mask = frame["policy_name"] == policy_name
        if not mask.any():
            continue
        q_col = next((col for col in candidate_cols if col in frame.columns), None)
        if q_col is None:
            continue
        frame.loc[mask, "q_value"] = frame.loc[mask, q_col].astype(float) * frame.loc[mask, "npv"].astype(float)
    frame["dr_row_value"] = frame["q_value"] + frame["weight"] * (frame["observed_value"] - frame["q_value"])
    return frame


def _safe_snips(total_weighted_value: float, total_weight: float, n_slots: int) -> float:
    if total_weight <= 0:
        return float("nan")
    return float(total_weighted_value / total_weight * n_slots)


def _effective_sample_size(weights: pd.Series) -> float:
    total = float(weights.sum())
    squared = float((weights**2).sum())
    if squared <= 0:
        return 0.0
    return total * total / squared


def summarize_policy_eval_frame(eval_frame: pd.DataFrame, n_slots: int = 3) -> pd.DataFrame:
    """Посчитать IPS/SNIPS/ESS и истинные синтетические значения для стратегий."""
    n_units = eval_frame[UNIT_COLUMNS].drop_duplicates().shape[0]
    rows = []

    for policy_name, policy_frame in eval_frame.groupby("policy_name", sort=False):
        total_weighted_value = float(policy_frame["weighted_value"].sum())
        total_weight = float(policy_frame["weight"].sum())
        matched_rows = int(policy_frame["matched"].sum())
        selected_rows = int(len(policy_frame))
        if policy_frame["q_value"].notna().all():
            dr_value = float(policy_frame.groupby(UNIT_COLUMNS)["dr_row_value"].sum().mean())
        else:
            dr_value = float("nan")

        rows.append(
            {
                "policy_name": policy_name,
                "n_units": n_units,
                "selected_rows": selected_rows,
                "matched_rows": matched_rows,
                "matched_share": matched_rows / selected_rows if selected_rows else 0.0,
                "ips_value": total_weighted_value / n_units,
                "snips_value": _safe_snips(total_weighted_value, total_weight, n_slots),
                "dr_value": dr_value,
                "ess": _effective_sample_size(policy_frame["weight"]),
                "true_shown_value": float(
                    policy_frame.groupby(UNIT_COLUMNS)["true_shown_value"].sum().mean()
                ),
                "true_extra_npv_value": float(
                    policy_frame.groupby(UNIT_COLUMNS)["true_extra_npv_value"].sum().mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("ips_value", ascending=False).reset_index(drop=True)


def _valid_set_count(unit_frame: pd.DataFrame, n_slots: int) -> int:
    available = unit_frame.loc[unit_frame["is_available"], ["category_id", "group_id"]]
    count = 0
    for rows in combinations(available.itertuples(index=False), n_slots):
        groups = [row.group_id for row in rows]
        if len(set(groups)) == len(groups):
            count += 1
    return count


def make_set_policy_eval_frame(
    assignments: pd.DataFrame,
    policy_assignments: pd.DataFrame,
    n_slots: int = 3,
) -> pd.DataFrame:
    """Соединить выбранные стратегией top-N наборы с randomized логом.

    Category-level IPS использует совпадения отдельных категорий. Set-level IPS
    строже: совпадение засчитывается только когда стратегия выбрала тот же
    unordered top-N набор, который был случайно показан в логе.
    """
    category_frame = make_policy_eval_frame(assignments, policy_assignments)

    valid_counts = (
        assignments.groupby(UNIT_COLUMNS, sort=False)[["category_id", "group_id", "is_available"]]
        .apply(lambda frame: _valid_set_count(frame, n_slots=n_slots))
        .rename("valid_set_count")
        .reset_index()
    )
    logged = (
        assignments.loc[assignments["shown"]]
        .sort_values(UNIT_COLUMNS + ["category_id"])
        .groupby(UNIT_COLUMNS, as_index=False)
        .agg(
            logged_set=("category_id", lambda values: tuple(sorted(int(v) for v in values))),
            observed_set_value=("utilized", "size"),
        )
    )
    logged_value = (
        assignments.loc[assignments["shown"]]
        .assign(observed_value=lambda frame: frame["utilized"].astype(float) * frame["npv"].astype(float))
        .groupby(UNIT_COLUMNS, as_index=False)["observed_value"]
        .sum()
        .rename(columns={"observed_value": "observed_set_value"})
    )
    logged = logged.drop(columns=["observed_set_value"]).merge(logged_value, on=UNIT_COLUMNS, how="left")
    logged = logged.merge(valid_counts, on=UNIT_COLUMNS, how="left")
    logged["set_propensity"] = 1.0 / logged["valid_set_count"].clip(lower=1)

    def sum_or_nan(values: pd.Series) -> float:
        return float("nan") if values.isna().any() else float(values.sum())

    policy_units = (
        category_frame.sort_values(["policy_name"] + UNIT_COLUMNS + ["category_id"])
        .groupby(["policy_name"] + UNIT_COLUMNS, as_index=False)
        .agg(
            split=("split", "first"),
            policy_set=("category_id", lambda values: tuple(sorted(int(v) for v in values))),
            selected_rows=("category_id", "size"),
            q_set_value=("q_value", sum_or_nan),
            true_shown_set_value=("true_shown_value", "sum"),
            true_extra_npv_set_value=("true_extra_npv_value", "sum"),
        )
    )
    frame = policy_units.merge(logged, on=UNIT_COLUMNS, how="left", validate="many_to_one")
    frame["set_matched"] = frame["policy_set"] == frame["logged_set"]
    frame["set_weight"] = np.where(frame["set_matched"], 1.0 / frame["set_propensity"].clip(lower=1e-12), 0.0)
    frame["set_weighted_value"] = frame["set_weight"] * frame["observed_set_value"]
    frame["set_dr_row_value"] = frame["q_set_value"] + frame["set_weight"] * (
        frame["observed_set_value"] - frame["q_set_value"]
    )
    return frame


def _safe_set_snips(total_weighted_value: float, total_weight: float) -> float:
    if total_weight <= 0:
        return float("nan")
    return float(total_weighted_value / total_weight)


def summarize_set_policy_eval_frame(set_eval_frame: pd.DataFrame) -> pd.DataFrame:
    n_units = set_eval_frame[UNIT_COLUMNS].drop_duplicates().shape[0]
    rows = []

    for policy_name, policy_frame in set_eval_frame.groupby("policy_name", sort=False):
        total_weighted_value = float(policy_frame["set_weighted_value"].sum())
        total_weight = float(policy_frame["set_weight"].sum())
        if policy_frame["q_set_value"].notna().all():
            set_dr_value = float(policy_frame.groupby(UNIT_COLUMNS)["set_dr_row_value"].sum().mean())
        else:
            set_dr_value = float("nan")

        rows.append(
            {
                "policy_name": policy_name,
                "set_n_units": n_units,
                "set_matched_units": int(policy_frame["set_matched"].sum()),
                "set_matched_share": float(policy_frame["set_matched"].mean()),
                "set_ips_value": total_weighted_value / n_units,
                "set_snips_value": _safe_set_snips(total_weighted_value, total_weight),
                "set_dr_value": set_dr_value,
                "set_ess": _effective_sample_size(policy_frame["set_weight"]),
                "set_true_shown_value": float(policy_frame["true_shown_set_value"].mean()),
                "set_true_extra_npv_value": float(policy_frame["true_extra_npv_set_value"].mean()),
            }
        )

    return pd.DataFrame(rows)


def _bootstrap_one_policy(
    policy_frame: pd.DataFrame,
    n_slots: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    unit_values = (
        policy_frame.groupby(UNIT_COLUMNS, as_index=False)
        .agg(weighted_value=("weighted_value", "sum"), weight=("weight", "sum"))
        .reset_index(drop=True)
    )
    n_units = len(unit_values)
    if n_units == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    ips_values = []
    snips_values = []
    weighted_value = unit_values["weighted_value"].to_numpy()
    weights = unit_values["weight"].to_numpy()

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_units, size=n_units)
        sample_weighted_value = float(weighted_value[idx].sum())
        sample_weight = float(weights[idx].sum())
        ips_values.append(sample_weighted_value / n_units)
        snips_values.append(_safe_snips(sample_weighted_value, sample_weight, n_slots))

    return (
        float(np.nanquantile(ips_values, 0.025)),
        float(np.nanquantile(ips_values, 0.975)),
        float(np.nanquantile(snips_values, 0.025)),
        float(np.nanquantile(snips_values, 0.975)),
    )


def add_bootstrap_intervals(
    summary: pd.DataFrame,
    eval_frame: pd.DataFrame,
    n_slots: int = 3,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for policy_name, policy_frame in eval_frame.groupby("policy_name", sort=False):
        ips_low, ips_high, snips_low, snips_high = _bootstrap_one_policy(
            policy_frame,
            n_slots=n_slots,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        rows.append(
            {
                "policy_name": policy_name,
                "ips_ci_low": ips_low,
                "ips_ci_high": ips_high,
                "snips_ci_low": snips_low,
                "snips_ci_high": snips_high,
            }
        )
    return summary.merge(pd.DataFrame(rows), on="policy_name", how="left")


def add_set_bootstrap_intervals(
    summary: pd.DataFrame,
    set_eval_frame: pd.DataFrame,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for policy_name, policy_frame in set_eval_frame.groupby("policy_name", sort=False):
        unit_values = (
            policy_frame.groupby(UNIT_COLUMNS, as_index=False)
            .agg(set_weighted_value=("set_weighted_value", "sum"), set_weight=("set_weight", "sum"))
            .reset_index(drop=True)
        )
        n_units = len(unit_values)
        if n_units == 0:
            rows.append(
                {
                    "policy_name": policy_name,
                    "set_ips_ci_low": float("nan"),
                    "set_ips_ci_high": float("nan"),
                    "set_snips_ci_low": float("nan"),
                    "set_snips_ci_high": float("nan"),
                }
            )
            continue

        ips_values = []
        snips_values = []
        weighted_value = unit_values["set_weighted_value"].to_numpy()
        weights = unit_values["set_weight"].to_numpy()
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_units, size=n_units)
            sample_weighted_value = float(weighted_value[idx].sum())
            sample_weight = float(weights[idx].sum())
            ips_values.append(sample_weighted_value / n_units)
            snips_values.append(_safe_set_snips(sample_weighted_value, sample_weight))

        snips_array = np.asarray(snips_values, dtype=float)
        if np.isfinite(snips_array).any():
            snips_low = float(np.nanquantile(snips_array, 0.025))
            snips_high = float(np.nanquantile(snips_array, 0.975))
        else:
            snips_low = float("nan")
            snips_high = float("nan")

        rows.append(
            {
                "policy_name": policy_name,
                "set_ips_ci_low": float(np.nanquantile(ips_values, 0.025)),
                "set_ips_ci_high": float(np.nanquantile(ips_values, 0.975)),
                "set_snips_ci_low": snips_low,
                "set_snips_ci_high": snips_high,
            }
        )

    return summary.merge(pd.DataFrame(rows), on="policy_name", how="left")


def evaluate_policies(
    assignments: pd.DataFrame,
    policy_assignments: pd.DataFrame,
    n_slots: int = 3,
    split: str | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Оценить стратегии на randomized логах.

    Если `split` задан, оценка проводится только на выбранном временном срезе.
    """
    assignments_eval = assignments.copy()
    policy_eval = policy_assignments.copy()
    if split is not None:
        assignments_eval = assignments_eval.loc[assignments_eval["split"] == split].copy()
        policy_eval = policy_eval.merge(
            assignments_eval[UNIT_COLUMNS].drop_duplicates(),
            on=UNIT_COLUMNS,
            how="inner",
        )

    eval_frame = make_policy_eval_frame(assignments_eval, policy_eval)
    summary = summarize_policy_eval_frame(eval_frame, n_slots=n_slots)
    category_summary = add_bootstrap_intervals(
        summary,
        eval_frame,
        n_slots=n_slots,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    set_eval_frame = make_set_policy_eval_frame(assignments_eval, policy_eval, n_slots=n_slots)
    set_summary = summarize_set_policy_eval_frame(set_eval_frame)
    set_summary = add_set_bootstrap_intervals(
        set_summary,
        set_eval_frame,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return category_summary.merge(set_summary, on="policy_name", how="left")
