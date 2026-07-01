"""Утилиты генерации синтетических данных.

Модуль создает публично безопасный датасет для CrossSell extra NPV:
пользователей, категории, исторические события, рандомизированные показы,
истинные потенциальные вероятности и последовательности для трансформера.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


GROUP_IDS = (
    "subscription",
    "insurance",
    "account",
    "investment",
    "credit",
    "partner",
)

HISTORY_FEATURE_COLUMNS = [
    "clicks_90d",
    "transactions_90d",
    "prior_shown_count",
    "prior_ignored_count",
    "prior_utilized_count",
    "app_actions_30d",
    "related_recency_days",
]

TRANSFORMER_SEQUENCE_FEATURE_COLUMNS = [
    "sequence_intent_score",
    "sequence_commit_count",
]

USER_FEATURE_COLUMNS = [
    "activity_level",
    "product_maturity",
    "recommendation_sensitivity",
    "offer_fatigue",
    "transaction_intensity",
    "app_engagement",
]

CATEGORY_FEATURE_COLUMNS = [
    "availability_rate",
    "base_popularity",
]

EVENT_FAMILIES = (
    "click",
    "transaction",
    "app_action",
    "prior_offer",
    "product_lifecycle",
)

EVENT_CATEGORICAL_COLUMNS = (
    "event_family",
    "event_type",
    "entity_type",
    "entity_id",
    "category_id",
    "group_id",
    "amount_bucket",
    "channel",
)

SEQUENCE_INTENT_EVENT_TYPES = ("intent_probe", "intent_compare", "intent_commit")
SEQUENCE_INTENT_WEIGHTS = {
    "intent_probe": 0.70,
    "intent_compare": 1.15,
    "intent_commit": 1.75,
}


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    n_users: int = 900
    n_categories: int = 18
    n_periods: int = 8
    min_history_periods: int = 2
    period_days: int = 30
    data_lag_days: int = 3
    target_window_days: int = 21
    n_slots: int = 3
    latent_dim: int = 8
    max_seq_len: int = 64


@dataclass
class SyntheticDataset:
    users: pd.DataFrame
    categories: pd.DataFrame
    events: pd.DataFrame
    assignments: pd.DataFrame


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def split_for_period(period_id: int, config: SyntheticConfig) -> str:
    periods = list(range(config.min_history_periods, config.n_periods))
    if period_id not in periods:
        return "history"

    rank = periods.index(period_id)
    n = len(periods)
    if n == 1:
        return "train"
    if rank == n - 1:
        return "test"
    if n >= 3 and rank == n - 2:
        return "val"
    if n >= 4 and rank == n - 3:
        return "calib"
    return "train"


def generate_users(config: SyntheticConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    latent = rng.normal(0.0, 1.0, size=(config.n_users, config.latent_dim))

    users = pd.DataFrame({"user_id": np.arange(config.n_users)})
    users["activity_level"] = sigmoid(0.90 * latent[:, 0])
    users["product_maturity"] = sigmoid(0.80 * latent[:, 1] + 0.20 * latent[:, 0])
    users["recommendation_sensitivity"] = sigmoid(1.10 * latent[:, 2] - 0.25 * latent[:, 3])
    users["offer_fatigue"] = sigmoid(0.95 * latent[:, 3] - 0.25 * latent[:, 2])
    users["transaction_intensity"] = sigmoid(0.75 * latent[:, 4] + 0.30 * latent[:, 0])
    users["app_engagement"] = sigmoid(0.80 * latent[:, 5] + 0.25 * latent[:, 0])

    pref_logits = rng.normal(0.0, 0.8, size=(config.n_users, len(GROUP_IDS)))
    pref_logits += users["product_maturity"].to_numpy()[:, None] * rng.normal(
        0.0, 0.35, size=(1, len(GROUP_IDS))
    )
    pref = np.exp(pref_logits - pref_logits.max(axis=1, keepdims=True))
    pref /= pref.sum(axis=1, keepdims=True)
    for idx, group_id in enumerate(GROUP_IDS):
        users[f"pref_{group_id}"] = pref[:, idx]

    for dim in range(config.latent_dim):
        users[f"user_emb_{dim}"] = latent[:, dim]

    return users


def generate_categories(config: SyntheticConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 11)
    group_ids = [GROUP_IDS[i % len(GROUP_IDS)] for i in range(config.n_categories)]
    rng.shuffle(group_ids)

    latent = rng.normal(0.0, 1.0, size=(config.n_categories, config.latent_dim))
    npv_raw = rng.lognormal(mean=3.15, sigma=0.35, size=config.n_categories)
    npv = 20.0 + 80.0 * (npv_raw - npv_raw.min()) / (npv_raw.max() - npv_raw.min())

    categories = pd.DataFrame(
        {
            "category_id": np.arange(config.n_categories),
            "group_id": group_ids,
            "npv": np.round(npv, 3),
            "availability_rate": rng.uniform(0.62, 0.92, size=config.n_categories),
            "base_popularity": rng.normal(0.0, 0.55, size=config.n_categories),
            "treatment_sensitivity": rng.uniform(0.04, 0.18, size=config.n_categories),
        }
    )

    for dim in range(config.latent_dim):
        categories[f"cat_emb_{dim}"] = latent[:, dim]

    return categories


def _choice_by_group_preference(
    rng: np.random.Generator,
    user_row,
    categories: pd.DataFrame,
) -> pd.Series:
    group_probs = np.array([getattr(user_row, f"pref_{t}") for t in GROUP_IDS], dtype=float)
    group_probs /= group_probs.sum()
    selected_group = rng.choice(GROUP_IDS, p=group_probs)
    candidates = categories.loc[categories["group_id"] == selected_group]
    if candidates.empty:
        candidates = categories
    return candidates.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]


def _background_event_record(
    user_id: int,
    period_id: int,
    event_time: int,
    family: str,
    event_type: str,
    entity_type: str,
    entity_id,
    category_id,
    group_id,
    amount_bucket,
    channel: str,
    event_value: float,
) -> dict:
    return {
        "user_id": int(user_id),
        "event_time": int(event_time),
        "event_family": family,
        "event_type": event_type,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "category_id": category_id,
        "group_id": group_id,
        "amount_bucket": amount_bucket,
        "channel": channel,
        "event_value": float(event_value),
        "period_id": int(period_id),
    }


def generate_background_events(
    users: pd.DataFrame,
    categories: pd.DataFrame,
    config: SyntheticConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 23)
    records = []
    channels = np.array(["app", "web", "push", "marketplace", "other"])

    for user_row in users.itertuples(index=False):
        base_events = 1.5 + 6.0 * user_row.activity_level + 2.5 * user_row.app_engagement
        family_probs = np.array(
            [
                0.18 + 0.16 * user_row.recommendation_sensitivity,
                0.26 + 0.24 * user_row.transaction_intensity,
                0.22 + 0.25 * user_row.app_engagement,
                0.02,
                0.06 + 0.10 * user_row.product_maturity,
            ]
        )
        family_probs = family_probs / family_probs.sum()

        for period_id in range(config.n_periods):
            n_events = int(rng.poisson(base_events))
            for _ in range(n_events):
                family = str(rng.choice(EVENT_FAMILIES, p=family_probs))
                event_time = period_id * config.period_days + int(rng.integers(0, config.period_days))
                category_row = _choice_by_group_preference(rng, user_row, categories)
                channel = str(rng.choice(channels))

                category_id = int(category_row["category_id"])
                group_id = str(category_row["group_id"])
                amount_bucket = None
                entity_type = "category"
                entity_id = f"category_{category_id}"
                event_value = float(rng.uniform(0.2, 1.0))

                if family == "click":
                    event_type = str(
                        rng.choice(
                            ["category_card_open", "details_view", "cta_click", "offer_dismiss"],
                            p=[0.38, 0.30, 0.17, 0.15],
                        )
                    )
                elif family == "transaction":
                    event_type = str(
                        rng.choice(
                            ["merchant_group_spend", "recurring_payment", "income_bucket", "saving_activity"],
                            p=[0.52, 0.23, 0.12, 0.13],
                        )
                    )
                    entity_type = "merchant_group"
                    entity_id = f"merchant_group_{int(rng.integers(0, 24))}"
                    amount_bucket = str(rng.choice(["low", "medium", "high"], p=[0.55, 0.32, 0.13]))
                    event_value = float({"low": 0.25, "medium": 0.55, "high": 0.90}[amount_bucket])
                elif family == "app_action":
                    event_type = str(
                        rng.choice(
                            ["screen_open", "search_used", "product_hub_visit", "document_open", "support_action"],
                            p=[0.42, 0.16, 0.23, 0.11, 0.08],
                        )
                    )
                    entity_type = "screen"
                    entity_id = f"screen_group_{int(rng.integers(0, 12))}"
                    if rng.random() < 0.55:
                        category_id = None
                        group_id = None
                elif family == "product_lifecycle":
                    event_type = str(
                        rng.choice(
                            ["product_opened", "product_closed", "service_started", "eligibility_changed"],
                            p=[0.38, 0.14, 0.26, 0.22],
                        )
                    )
                    entity_type = "product"
                    entity_id = f"synthetic_product_{category_id}"
                else:
                    event_type = "background_offer_signal"

                records.append(
                    _background_event_record(
                        user_id=user_row.user_id,
                        period_id=period_id,
                        event_time=event_time,
                        family=family,
                        event_type=event_type,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        category_id=category_id,
                        group_id=group_id,
                        amount_bucket=amount_bucket,
                        channel=channel,
                        event_value=event_value,
                        )
                )

            intent_prob = np.clip(
                0.05
                + 0.42 * user_row.recommendation_sensitivity
                + 0.22 * user_row.app_engagement
                - 0.18 * user_row.offer_fatigue,
                0.04,
                0.72,
            )
            n_intent_sequences = int(rng.random() < intent_prob)
            n_intent_sequences += int(rng.random() < 0.20 * user_row.activity_level)
            for _ in range(n_intent_sequences):
                category_row = _choice_by_group_preference(rng, user_row, categories)
                category_id = int(category_row["category_id"])
                group_id = str(category_row["group_id"])
                base_time = period_id * config.period_days + int(rng.integers(4, max(5, config.period_days - 8)))
                for offset, event_type in enumerate(SEQUENCE_INTENT_EVENT_TYPES):
                    records.append(
                        _background_event_record(
                            user_id=user_row.user_id,
                            period_id=period_id,
                            event_time=base_time + offset,
                            family="app_action",
                            event_type=event_type,
                            entity_type="category",
                            entity_id=f"category_{category_id}",
                            category_id=category_id,
                            group_id=group_id,
                            amount_bucket=None,
                            channel="app",
                            event_value=SEQUENCE_INTENT_WEIGHTS[event_type],
                        )
                    )

    return pd.DataFrame.from_records(records).sort_values(["user_id", "event_time"]).reset_index(drop=True)


def _empty_history_grid(
    users: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [users["user_id"].to_numpy(), categories["category_id"].to_numpy()],
        names=["user_id", "category_id"],
    ).to_frame(index=False)
    return grid


def build_history_features(
    events: pd.DataFrame,
    users: pd.DataFrame,
    categories: pd.DataFrame,
    period_id: int,
    config: SyntheticConfig,
) -> pd.DataFrame:
    cutoff = period_id * config.period_days - config.data_lag_days
    hist = events.loc[events["event_time"] <= cutoff].copy()
    recent_30 = hist["event_time"] > cutoff - 30
    recent_90 = hist["event_time"] > cutoff - 90

    grid = _empty_history_grid(users, categories)
    features = grid.copy()

    def merge_count(mask: pd.Series, name: str) -> None:
        count = (
            hist.loc[mask & hist["category_id"].notna()]
            .groupby(["user_id", "category_id"])
            .size()
            .rename(name)
            .reset_index()
        )
        nonlocal features
        features = features.merge(count, on=["user_id", "category_id"], how="left")

    merge_count((hist["event_family"] == "click") & recent_90, "clicks_90d")
    merge_count((hist["event_family"] == "transaction") & recent_90, "transactions_90d")
    merge_count((hist["event_type"] == "offer_shown"), "prior_shown_count")
    merge_count((hist["event_type"] == "offer_ignored"), "prior_ignored_count")
    merge_count((hist["event_type"] == "offer_utilized"), "prior_utilized_count")

    user_app = (
        hist.loc[(hist["event_family"] == "app_action") & recent_30]
        .groupby("user_id")
        .size()
        .rename("app_actions_30d")
        .reset_index()
    )
    features = features.merge(user_app, on="user_id", how="left")

    last_related = (
        hist.loc[hist["category_id"].notna()]
        .groupby(["user_id", "category_id"])["event_time"]
        .max()
        .rename("last_related_event_time")
        .reset_index()
    )
    features = features.merge(last_related, on=["user_id", "category_id"], how="left")
    features["related_recency_days"] = cutoff - features["last_related_event_time"]
    features["related_recency_days"] = features["related_recency_days"].fillna(999.0).clip(lower=0)
    features = features.drop(columns=["last_related_event_time"])

    intent_mask = (
        hist["event_type"].isin(SEQUENCE_INTENT_EVENT_TYPES)
        & hist["category_id"].notna()
        & (hist["event_time"] > cutoff - 75)
    )
    intent_events = hist.loc[intent_mask, ["user_id", "category_id", "event_time", "event_type"]].copy()
    if not intent_events.empty:
        intent_events["intent_weight"] = intent_events["event_type"].map(SEQUENCE_INTENT_WEIGHTS).astype(float)
        intent_events["intent_weight"] *= np.exp(-(cutoff - intent_events["event_time"].astype(float)) / 28.0)
        intent_score = (
            intent_events.groupby(["user_id", "category_id"])["intent_weight"]
            .sum()
            .rename("sequence_intent_score")
            .reset_index()
        )
        intent_commit = (
            intent_events.loc[intent_events["event_type"] == "intent_commit"]
            .groupby(["user_id", "category_id"])
            .size()
            .rename("sequence_commit_count")
            .reset_index()
        )
        features = features.merge(intent_score, on=["user_id", "category_id"], how="left")
        features = features.merge(intent_commit, on=["user_id", "category_id"], how="left")
    else:
        features["sequence_intent_score"] = 0.0
        features["sequence_commit_count"] = 0.0

    for col in [
        "clicks_90d",
        "transactions_90d",
        "prior_shown_count",
        "prior_ignored_count",
        "prior_utilized_count",
        "app_actions_30d",
        "sequence_intent_score",
        "sequence_commit_count",
    ]:
        features[col] = features[col].fillna(0.0).astype(float)

    return features


def _add_group_preference(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["group_pref"] = 0.0
    for group_id in GROUP_IDS:
        mask = out["group_id"] == group_id
        out.loc[mask, "group_pref"] = out.loc[mask, f"pref_{group_id}"].astype(float)
    return out


def _valid_sets_for_user(user_frame: pd.DataFrame, n_slots: int) -> list[tuple[int, ...]]:
    available = user_frame.loc[user_frame["is_available"], ["category_id", "group_id"]]
    valid_sets: list[tuple[int, ...]] = []
    for rows in combinations(available.itertuples(index=False), n_slots):
        types = [row.group_id for row in rows]
        if len(set(types)) == len(types):
            valid_sets.append(tuple(int(row.category_id) for row in rows))
    return valid_sets


def _force_valid_availability(
    period_frame: pd.DataFrame,
    categories: pd.DataFrame,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = period_frame.copy()
    categories_by_group = {
        group_id: categories.loc[categories["group_id"] == group_id, "category_id"].to_numpy()
        for group_id in GROUP_IDS
    }

    for user_id, user_frame in out.groupby("user_id", sort=False):
        if _valid_sets_for_user(user_frame, config.n_slots):
            continue
        chosen_groups = rng.choice(GROUP_IDS, size=config.n_slots, replace=False)
        forced_category_ids = []
        for group_id in chosen_groups:
            options = categories_by_group[group_id]
            forced_category_ids.append(int(rng.choice(options)))
        mask = (out["user_id"] == user_id) & (out["category_id"].isin(forced_category_ids))
        out.loc[mask, "is_available"] = True

    return out


def _score_true_outcomes(period_frame: pd.DataFrame, period_id: int, config: SyntheticConfig) -> pd.DataFrame:
    out = _add_group_preference(period_frame)
    emb_dot = np.zeros(len(out))
    for dim in range(config.latent_dim):
        emb_dot += out[f"user_emb_{dim}"].to_numpy() * out[f"cat_emb_{dim}"].to_numpy()
    emb_dot = emb_dot / np.sqrt(config.latent_dim)

    recent_interest = np.log1p(out["clicks_90d"].to_numpy())
    transaction_signal = np.log1p(out["transactions_90d"].to_numpy())
    app_signal = np.log1p(out["app_actions_30d"].to_numpy())
    recency_signal = np.exp(-out["related_recency_days"].to_numpy() / 45.0)
    sequence_intent_raw = (
        out["sequence_intent_score"].to_numpy()
        if "sequence_intent_score" in out
        else np.zeros(len(out), dtype=float)
    )
    sequence_commit_raw = (
        out["sequence_commit_count"].to_numpy()
        if "sequence_commit_count" in out
        else np.zeros(len(out), dtype=float)
    )
    sequence_intent = np.log1p(sequence_intent_raw)
    sequence_commit = np.log1p(sequence_commit_raw)
    prior_shown = out["prior_shown_count"].to_numpy()
    prior_ignored = out["prior_ignored_count"].to_numpy()
    prior_utilized = out["prior_utilized_count"].to_numpy()

    seasonal = 0.10 * np.sin(period_id / 2.0)
    base_logit = (
        -3.15
        + 1.75 * out["group_pref"].to_numpy()
        + 0.80 * out["product_maturity"].to_numpy()
        + 0.55 * out["activity_level"].to_numpy()
        + 0.38 * transaction_signal
        + 0.48 * recent_interest
        + 0.20 * app_signal
        + 0.10 * sequence_intent
        + 0.06 * sequence_commit
        + 0.24 * emb_dot
        + 0.30 * out["base_popularity"].to_numpy()
        + 0.42 * recency_signal
        + 0.28 * prior_utilized
        - 0.18 * prior_ignored
        + seasonal
    )
    p0 = sigmoid(base_logit)

    response_headroom = np.clip(0.98 - p0, 0.02, 0.98)
    # Эффект показа (uplift) намеренно разведён с базовым откликом p0: он опирается
    # в основном на treatment-специфичные драйверы (склонность реагировать на показ,
    # чувствительность категории к воздействию, недавние intent-события из истории) и
    # отрицательно связан с p0. Драйверы базового отклика (group_pref, recent_interest,
    # app_signal, recency) в эффект почти не входят, поэтому ранжирование по extra NPV
    # отличается от ранжирования по отклику и демонстрирует свой смысл.
    effect_logit = (
        -0.55
        + 3.00 * out["recommendation_sensitivity"].to_numpy()
        + 4.10 * out["treatment_sensitivity"].to_numpy()
        + 2.90 * sequence_intent
        + 1.45 * sequence_commit
        - 2.20 * out["offer_fatigue"].to_numpy()
        + 0.15 * out["group_pref"].to_numpy()
        - 0.45 * prior_shown
        - 0.88 * prior_ignored
        - 1.30 * p0
    )
    positive_delta = response_headroom * (0.03 + 0.52 * sigmoid(effect_logit))
    negative_fatigue = 0.030 * np.clip(prior_ignored, 0, 3) + 0.018 * np.clip(prior_shown - 2, 0, 4)
    delta = np.clip(positive_delta - negative_fatigue, -0.055, 0.36)

    out["p0_true"] = np.where(out["is_available"], np.clip(p0, 0.001, 0.94), 0.0)
    out["p1_true"] = np.where(out["is_available"], np.clip(p0 + delta, 0.001, 0.98), 0.0)
    out["delta_p_true"] = out["p1_true"] - out["p0_true"]
    out["extra_npv_true"] = out["npv"] * out["delta_p_true"]
    return out


def _assign_random_sets(
    period_frame: pd.DataFrame,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = period_frame.copy()
    out["shown"] = False
    out["show_propensity"] = 0.0

    for user_id, user_frame in out.groupby("user_id", sort=False):
        valid_sets = _valid_sets_for_user(user_frame, config.n_slots)
        if not valid_sets:
            raise ValueError(f"Нет допустимых наборов для user_id={user_id}")

        selected = set(valid_sets[int(rng.integers(0, len(valid_sets)))])
        counts = {category_id: 0 for category_id in user_frame["category_id"].astype(int)}
        for valid_set in valid_sets:
            for category_id in valid_set:
                counts[int(category_id)] += 1

        user_mask = out["user_id"] == user_id
        for category_id, count in counts.items():
            category_mask = user_mask & (out["category_id"] == category_id)
            out.loc[category_mask, "show_propensity"] = count / len(valid_sets)
        out.loc[user_mask & out["category_id"].isin(selected), "shown"] = True

    out["propensity"] = np.where(out["shown"], out["show_propensity"], 1.0 - out["show_propensity"])
    out["propensity"] = out["propensity"].clip(1e-6, 1.0)
    return out


def _make_prior_offer_events(
    shown_rows: pd.DataFrame,
    period_id: int,
    config: SyntheticConfig,
) -> list[dict]:
    decision_time = period_id * config.period_days
    records = []
    for row in shown_rows.itertuples(index=False):
        records.append(
            _background_event_record(
                user_id=row.user_id,
                period_id=period_id,
                event_time=decision_time + 1,
                family="prior_offer",
                event_type="offer_shown",
                entity_type="category",
                entity_id=f"category_{int(row.category_id)}",
                category_id=int(row.category_id),
                group_id=row.group_id,
                amount_bucket=None,
                channel="marketplace",
                event_value=1.0,
            )
        )
        records.append(
            _background_event_record(
                user_id=row.user_id,
                period_id=period_id,
                event_time=decision_time + config.target_window_days,
                family="prior_offer",
                event_type="offer_utilized" if row.utilized else "offer_ignored",
                entity_type="category",
                entity_id=f"category_{int(row.category_id)}",
                category_id=int(row.category_id),
                group_id=row.group_id,
                amount_bucket=None,
                channel="marketplace",
                event_value=float(row.utilized),
            )
        )
    return records


def generate_assignments(
    users: pd.DataFrame,
    categories: pd.DataFrame,
    events: pd.DataFrame,
    config: SyntheticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed + 101)
    event_records = events.to_dict("records")
    assignment_frames = []

    for period_id in range(config.min_history_periods, config.n_periods):
        current_events = pd.DataFrame.from_records(event_records)
        history_features = build_history_features(current_events, users, categories, period_id, config)

        frame = history_features.merge(users, on="user_id", how="left").merge(
            categories, on="category_id", how="left"
        )
        frame = _add_group_preference(frame)

        availability_score = (
            frame["availability_rate"].to_numpy()
            + 0.16 * frame["product_maturity"].to_numpy()
            + 0.20 * frame["group_pref"].to_numpy()
            - 0.18 * np.clip(frame["prior_utilized_count"].to_numpy(), 0, 3)
        )
        availability_prob = np.clip(availability_score, 0.08, 0.98)
        frame["is_available"] = rng.random(len(frame)) < availability_prob
        frame = _force_valid_availability(frame, categories, config, rng)
        frame = _score_true_outcomes(frame, period_id, config)
        frame = _assign_random_sets(frame, config, rng)

        observed_prob = np.where(frame["shown"], frame["p1_true"], frame["p0_true"])
        frame["utilized"] = rng.binomial(1, observed_prob).astype(int)
        frame.loc[~frame["is_available"], "utilized"] = 0
        frame["period_id"] = period_id
        frame["decision_time"] = period_id * config.period_days
        frame["split"] = split_for_period(period_id, config)

        keep_cols = [
            "user_id",
            "period_id",
            "split",
            "decision_time",
            "category_id",
            "group_id",
            "npv",
            "is_available",
            "shown",
            "show_propensity",
            "propensity",
            "utilized",
            "p0_true",
            "p1_true",
            "delta_p_true",
            "extra_npv_true",
        ]
        model_feature_cols = (
            HISTORY_FEATURE_COLUMNS
            + USER_FEATURE_COLUMNS
            + CATEGORY_FEATURE_COLUMNS
            + ["group_pref"]
            + TRANSFORMER_SEQUENCE_FEATURE_COLUMNS
            + [f"pref_{group_id}" for group_id in GROUP_IDS]
            + [f"user_emb_{dim}" for dim in range(config.latent_dim)]
            + [f"cat_emb_{dim}" for dim in range(config.latent_dim)]
        )
        keep_cols = keep_cols + [col for col in model_feature_cols if col not in keep_cols]
        assignment_frames.append(frame[keep_cols].copy())

        shown_rows = frame.loc[frame["shown"], keep_cols].copy()
        event_records.extend(_make_prior_offer_events(shown_rows, period_id, config))

    assignments = pd.concat(assignment_frames, ignore_index=True)
    final_events = (
        pd.DataFrame.from_records(event_records)
        .sort_values(["user_id", "event_time", "event_type"])
        .reset_index(drop=True)
    )
    return assignments, final_events


def generate_synthetic_data(config: SyntheticConfig | None = None) -> SyntheticDataset:
    if config is None:
        config = SyntheticConfig()

    users = generate_users(config)
    categories = generate_categories(config)
    events = generate_background_events(users, categories, config)
    assignments, events = generate_assignments(users, categories, events, config)
    return SyntheticDataset(users=users, categories=categories, events=events, assignments=assignments)


def decision_units_from_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        assignments[["user_id", "period_id", "split", "decision_time"]]
        .drop_duplicates()
        .sort_values(["period_id", "user_id"])
        .reset_index(drop=True)
    )


def build_vocabularies(events: pd.DataFrame, columns: Iterable[str] = EVENT_CATEGORICAL_COLUMNS) -> dict[str, dict]:
    vocabs: dict[str, dict] = {}
    for col in columns:
        values = events[col].fillna("__null__").astype(str).sort_values().unique().tolist()
        vocabs[col] = {value: idx + 1 for idx, value in enumerate(values)}
    return vocabs


def recency_bucket(days: float) -> int:
    bins = np.array([0, 3, 7, 14, 30, 60, 90, 180, 365, 10_000], dtype=float)
    return int(np.searchsorted(bins, days, side="right"))


def build_transformer_sequence_table(
    events: pd.DataFrame,
    decision_units: pd.DataFrame,
    config: SyntheticConfig,
    vocabs: dict[str, dict] | None = None,
) -> pd.DataFrame:
    if vocabs is None:
        vocabs = build_vocabularies(events)

    records = []
    events_by_user = {user_id: df.sort_values("event_time") for user_id, df in events.groupby("user_id")}

    for unit in decision_units.itertuples(index=False):
        cutoff = int(unit.period_id) * config.period_days - config.data_lag_days
        hist = events_by_user.get(unit.user_id)
        if hist is None:
            hist = events.iloc[0:0]
        hist = hist.loc[hist["event_time"] <= cutoff].tail(config.max_seq_len)
        pad_len = config.max_seq_len - len(hist)

        for position in range(pad_len):
            records.append(
                {
                    "user_id": int(unit.user_id),
                    "period_id": int(unit.period_id),
                    "position": int(position),
                    "is_padding": 1,
                    "cutoff_time": int(cutoff),
                    "source_event_time": -1,
                    "recency_bucket": 0,
                    "event_value": 0.0,
                    **{f"{col}_id": 0 for col in EVENT_CATEGORICAL_COLUMNS},
                }
            )

        for offset, row in enumerate(hist.itertuples(index=False), start=pad_len):
            item = {
                "user_id": int(unit.user_id),
                "period_id": int(unit.period_id),
                "position": int(offset),
                "is_padding": 0,
                "cutoff_time": int(cutoff),
                "source_event_time": int(row.event_time),
                "recency_bucket": recency_bucket(cutoff - int(row.event_time)),
                "event_value": float(row.event_value),
            }
            row_dict = row._asdict()
            for col in EVENT_CATEGORICAL_COLUMNS:
                value = "__null__" if pd.isna(row_dict[col]) else str(row_dict[col])
                item[f"{col}_id"] = vocabs[col].get(value, 0)
            records.append(item)

    return pd.DataFrame.from_records(records)


def validate_synthetic_dataset(dataset: SyntheticDataset, config: SyntheticConfig) -> dict[str, bool]:
    assignments = dataset.assignments
    events = dataset.events

    shown_per_unit = assignments.groupby(["user_id", "period_id"])["shown"].sum()
    decision_units = decision_units_from_assignments(assignments)
    sequence = build_transformer_sequence_table(events, decision_units, config)
    non_padding = sequence["is_padding"] == 0
    no_future_events_in_history = bool(
        sequence.loc[non_padding, "source_event_time"].le(sequence.loc[non_padding, "cutoff_time"]).all()
    )
    fixed_sequence_length = bool(
        (sequence.groupby(["user_id", "period_id"])["position"].size() == config.max_seq_len).all()
    )
    non_negative_recency = bool((sequence.loc[non_padding, "recency_bucket"] >= 0).all())

    ceiling_positive = assignments["extra_npv_true"].max() > assignments["extra_npv_true"].mean()
    return {
        "fixed_slots": bool((shown_per_unit == config.n_slots).all()),
        "positive_propensity_for_shown": bool((assignments.loc[assignments["shown"], "show_propensity"] > 0).all()),
        "no_future_events_in_history": no_future_events_in_history,
        "fixed_sequence_length": fixed_sequence_length,
        "non_negative_recency_bucket": non_negative_recency,
        "ceiling_has_signal": bool(ceiling_positive),
    }
