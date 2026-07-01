"""Стратегии выбора категорий из преселекта.

На вход подается скор для каждой строки `user_id, period_id, category_id`.
Стратегия выбирает категории жадно: берет лучшую доступную категорию, удаляет
из рассмотрения все категории с тем же `group_id`, затем берет следующую.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


UNIT_COLUMNS = ["user_id", "period_id"]
BASE_COLUMNS = ["user_id", "period_id", "split", "category_id", "group_id", "is_available"]


def _required_columns(score_frame: pd.DataFrame, score_col: str) -> set[str]:
    return set(BASE_COLUMNS + [score_col])


def validate_score_frame(score_frame: pd.DataFrame, score_col: str) -> None:
    missing = _required_columns(score_frame, score_col) - set(score_frame.columns)
    if missing:
        raise ValueError(f"Не хватает колонок для построения стратегии: {sorted(missing)}")
    duplicated = score_frame.duplicated(UNIT_COLUMNS + ["category_id"])
    if duplicated.any():
        raise ValueError("В score_frame есть дубли на уровне user_id, period_id, category_id.")


def choose_top_categories_greedy(
    score_frame: pd.DataFrame,
    score_col: str,
    policy_name: str,
    n_slots: int = 3,
    descending: bool = True,
) -> pd.DataFrame:
    """Выбрать топ категорий по скору с учетом преселекта и уникальности group_id."""
    validate_score_frame(score_frame, score_col)

    rows = []
    available = score_frame.loc[score_frame["is_available"]].copy()

    for (user_id, period_id), unit_frame in available.groupby(UNIT_COLUMNS, sort=False):
        sorted_frame = unit_frame.sort_values(
            [score_col, "category_id"],
            ascending=[not descending, True],
            kind="mergesort",
        )
        used_groups = set()
        rank = 1

        for item in sorted_frame.itertuples(index=False):
            group_id = getattr(item, "group_id")
            if group_id in used_groups:
                continue

            used_groups.add(group_id)
            rows.append(
                {
                    "user_id": int(user_id),
                    "period_id": int(period_id),
                    "split": getattr(item, "split"),
                    "policy_name": policy_name,
                    "rank": rank,
                    "category_id": int(getattr(item, "category_id")),
                    "group_id": group_id,
                    "policy_score": float(getattr(item, score_col)),
                    "assigned": True,
                }
            )
            rank += 1
            if rank > n_slots:
                break

        if rank <= n_slots:
            raise ValueError(
                f"Недостаточно доступных group_id для user_id={user_id}, period_id={period_id}: "
                f"нужно {n_slots}, выбрано {rank - 1}."
            )

    return pd.DataFrame(rows).sort_values(UNIT_COLUMNS + ["rank"]).reset_index(drop=True)


def with_response_true_score(candidate_frame: pd.DataFrame) -> pd.DataFrame:
    return with_response_score(candidate_frame, probability_col="p1_true", output_col="response_true_score")


def with_response_score(
    candidate_frame: pd.DataFrame,
    probability_col: str,
    output_col: str = "response_score",
) -> pd.DataFrame:
    """Посчитать приоритизацию `вероятность утилизации * NPV`."""
    missing = {"npv", probability_col} - set(candidate_frame.columns)
    if missing:
        raise ValueError(f"Не хватает колонок для response-приоритизации: {sorted(missing)}")
    out = candidate_frame.copy()
    out[output_col] = out["npv"].astype(float) * out[probability_col].astype(float)
    return out


def with_extra_npv_score(
    candidate_frame: pd.DataFrame,
    p1_col: str,
    p0_col: str,
    output_col: str = "extra_npv_score",
) -> pd.DataFrame:
    """Посчитать приоритизацию `NPV * (p1 - p0)`."""
    missing = {"npv", p1_col, p0_col} - set(candidate_frame.columns)
    if missing:
        raise ValueError(f"Не хватает колонок для extra NPV-приоритизации: {sorted(missing)}")
    out = candidate_frame.copy()
    out[output_col] = out["npv"].astype(float) * (
        out[p1_col].astype(float) - out[p0_col].astype(float)
    )
    return out


def with_random_score(candidate_frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = candidate_frame.copy()
    out["random_score"] = rng.random(len(out))
    return out


def make_random_policy(candidate_frame: pd.DataFrame, n_slots: int = 3, seed: int = 42) -> pd.DataFrame:
    scored = with_random_score(candidate_frame, seed=seed)
    return choose_top_categories_greedy(scored, "random_score", "random", n_slots=n_slots)


def make_npv_policy(candidate_frame: pd.DataFrame, n_slots: int = 3) -> pd.DataFrame:
    return choose_top_categories_greedy(candidate_frame, "npv", "npv", n_slots=n_slots)


def make_response_policy(
    candidate_frame: pd.DataFrame,
    probability_col: str,
    policy_name: str,
    n_slots: int = 3,
) -> pd.DataFrame:
    scored = with_response_score(
        candidate_frame,
        probability_col=probability_col,
        output_col=f"{policy_name}_score",
    )
    return choose_top_categories_greedy(scored, f"{policy_name}_score", policy_name, n_slots=n_slots)


def make_extra_npv_policy(
    candidate_frame: pd.DataFrame,
    p1_col: str,
    p0_col: str,
    policy_name: str,
    n_slots: int = 3,
) -> pd.DataFrame:
    scored = with_extra_npv_score(
        candidate_frame,
        p1_col=p1_col,
        p0_col=p0_col,
        output_col=f"{policy_name}_score",
    )
    return choose_top_categories_greedy(scored, f"{policy_name}_score", policy_name, n_slots=n_slots)


def make_response_true_policy(candidate_frame: pd.DataFrame, n_slots: int = 3) -> pd.DataFrame:
    return make_response_policy(
        candidate_frame,
        probability_col="p1_true",
        policy_name="response_true",
        n_slots=n_slots,
    )


def make_synthetic_ceiling_policy(candidate_frame: pd.DataFrame, n_slots: int = 3) -> pd.DataFrame:
    return make_extra_npv_policy(
        candidate_frame,
        p1_col="p1_true",
        p0_col="p0_true",
        policy_name="synthetic_ceiling",
        n_slots=n_slots,
    )


def make_policy_family(candidate_frame: pd.DataFrame, n_slots: int = 3, seed: int = 42) -> pd.DataFrame:
    """Построить базовый набор стратегий для проверки пайплайна до обучения моделей."""
    policies = [
        make_random_policy(candidate_frame, n_slots=n_slots, seed=seed),
        make_npv_policy(candidate_frame, n_slots=n_slots),
        make_response_true_policy(candidate_frame, n_slots=n_slots),
        make_synthetic_ceiling_policy(candidate_frame, n_slots=n_slots),
    ]
    return pd.concat(policies, ignore_index=True)


def make_model_policy_family(candidate_frame: pd.DataFrame, n_slots: int = 3) -> pd.DataFrame:
    """Построить политики на скорингах обученных моделей, если нужные колонки есть."""
    specs = [
        ("p_util_hat", None, "response_model"),
        ("p1_hat", "p0_hat", "extra_npv_model"),
        ("transformer_p_util_hat", None, "response_transformer"),
        ("transformer_p1_hat", "transformer_p0_hat", "extra_npv_transformer"),
        ("stacked_p_util_hat", None, "response_stacked"),
        ("stacked_p1_hat", "stacked_p0_hat", "extra_npv_stacked"),
    ]
    policies = []
    for p1_col, p0_col, policy_name in specs:
        if p1_col not in candidate_frame.columns:
            continue
        if p0_col is None:
            policies.append(
                make_response_policy(
                    candidate_frame,
                    probability_col=p1_col,
                    policy_name=policy_name,
                    n_slots=n_slots,
                )
            )
        elif p0_col in candidate_frame.columns:
            policies.append(
                make_extra_npv_policy(
                    candidate_frame,
                    p1_col=p1_col,
                    p0_col=p0_col,
                    policy_name=policy_name,
                    n_slots=n_slots,
                )
            )

    if not policies:
        return pd.DataFrame(
            columns=[
                "user_id",
                "period_id",
                "split",
                "policy_name",
                "rank",
                "category_id",
                "group_id",
                "policy_score",
                "assigned",
            ]
        )
    return pd.concat(policies, ignore_index=True)


def policy_slot_counts(policy_assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        policy_assignments.groupby(["policy_name", "user_id", "period_id"], as_index=False)
        .agg(n_slots=("category_id", "size"), n_groups=("group_id", "nunique"))
        .sort_values(["policy_name", "period_id", "user_id"])
        .reset_index(drop=True)
    )
