"""Модели отклика, uplift, трансформерный энкодер и stacking."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.data import (
    EVENT_CATEGORICAL_COLUMNS,
    SyntheticConfig,
    TRANSFORMER_SEQUENCE_FEATURE_COLUMNS,
    build_transformer_sequence_table,
    build_vocabularies,
    decision_units_from_assignments,
)


TARGET_COLUMNS = {
    "utilized",
    "shown",
    "propensity",
    "show_propensity",
    "p0_true",
    "p1_true",
    "delta_p_true",
    "extra_npv_true",
}

ID_COLUMNS = {"user_id", "decision_time"}
PREDICTION_COLUMNS = {
    "p_util_hat",
    "p1_hat",
    "p0_hat",
    "uplift_hat",
    "response_model_score",
    "extra_npv_model_score",
    "transformer_p_util_hat",
    "transformer_p1_hat",
    "transformer_p0_hat",
    "transformer_uplift_hat",
    "stacked_p_util_hat",
    "stacked_p1_hat",
    "stacked_p0_hat",
    "stacked_uplift_hat",
}
CATEGORICAL_FEATURE_COLUMNS = ["group_id"]
TRANSFORMER_SCORE_COLUMNS = [
    "transformer_p_util_hat",
    "transformer_p1_hat",
    "transformer_p0_hat",
    "transformer_uplift_hat",
]
TREATMENT_FEATURE_COLUMN = "shown_feature"
TREATMENT_INTERACTION_PREFIX = f"{TREATMENT_FEATURE_COLUMN}_x_"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


@dataclass
class DecisionStump:
    feature_idx: int
    threshold: float
    left_value: float
    right_value: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(x[:, self.feature_idx] <= self.threshold, self.left_value, self.right_value)


class SimpleGradientBoostingClassifier:
    """Небольшой градиентный бустинг на decision stumps для бинарной классификации."""

    def __init__(
        self,
        n_estimators: int = 220,
        learning_rate: float = 0.055,
        max_thresholds: int = 18,
        min_leaf: int = 14,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_thresholds = max_thresholds
        self.min_leaf = min_leaf
        self.random_state = random_state
        self.base_logit = 0.0
        self.stumps: list[DecisionStump] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SimpleGradientBoostingClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        p = np.clip(y.mean(), 1e-4, 1.0 - 1e-4)
        self.base_logit = float(np.log(p / (1.0 - p)))
        raw = np.full(len(y), self.base_logit, dtype=float)
        self.stumps = []

        for _ in range(self.n_estimators):
            residual = y - sigmoid(raw)
            stump = self._fit_stump(x, residual)
            update = stump.predict(x)
            raw += self.learning_rate * update
            self.stumps.append(stump)

        return self

    def _fit_stump(self, x: np.ndarray, residual: np.ndarray) -> DecisionStump:
        best_loss = np.inf
        best = DecisionStump(feature_idx=0, threshold=0.0, left_value=0.0, right_value=0.0)

        for feature_idx in range(x.shape[1]):
            values = x[:, feature_idx]
            if np.all(values == values[0]):
                continue
            quantiles = np.linspace(0.08, 0.92, self.max_thresholds)
            thresholds = np.unique(np.quantile(values, quantiles))
            for threshold in thresholds:
                left = values <= threshold
                right = ~left
                if left.sum() < self.min_leaf or right.sum() < self.min_leaf:
                    continue
                left_value = float(residual[left].mean())
                right_value = float(residual[right].mean())
                pred = np.where(left, left_value, right_value)
                loss = float(np.mean((residual - pred) ** 2))
                if loss < best_loss:
                    best_loss = loss
                    best = DecisionStump(
                        feature_idx=feature_idx,
                        threshold=float(threshold),
                        left_value=left_value,
                        right_value=right_value,
                    )

        if not np.isfinite(best_loss):
            return DecisionStump(feature_idx=0, threshold=0.0, left_value=float(residual.mean()), right_value=0.0)
        return best

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        raw = np.full(x.shape[0], self.base_logit, dtype=float)
        for stump in self.stumps:
            raw += self.learning_rate * stump.predict(x)
        return raw

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = sigmoid(self.predict_raw(x))
        return np.column_stack([1.0 - p1, p1])


class ConstantProbabilityModel:
    """Запасная модель для редких случаев, когда в обучении один класс."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, x) -> np.ndarray:
        p1 = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - p1, p1])


@dataclass
class FittedBinaryModel:
    model: object
    input_feature_columns: list[str]
    training_columns: list[str]


@dataclass
class ModelBundle:
    response_model: FittedBinaryModel
    treatment_model: FittedBinaryModel
    control_model: FittedBinaryModel
    learner_type: str = "s_learner"


@dataclass
class TransformerEncoder:
    vocabs: dict[str, dict]
    embedding_tables: dict[str, np.ndarray]
    recency_embedding: np.ndarray
    position_embedding: np.ndarray
    value_vector: np.ndarray
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    w_o: np.ndarray
    pool_query: np.ndarray
    head_w: np.ndarray
    head_b: float
    d_model: int
    is_trained: bool = False
    training_losses: list[float] | None = None


@dataclass
class TransformerBundle:
    encoder: TransformerEncoder
    response_model: FittedBinaryModel
    treatment_model: FittedBinaryModel
    control_model: FittedBinaryModel
    transformer_feature_columns: list[str]


def infer_feature_columns(
    frame: pd.DataFrame,
    include_prediction_features: bool = False,
) -> list[str]:
    candidates = []
    for col in frame.columns:
        if col in TARGET_COLUMNS or col in ID_COLUMNS:
            continue
        if col in TRANSFORMER_SEQUENCE_FEATURE_COLUMNS:
            continue
        if col == "split":
            continue
        if not include_prediction_features and col in PREDICTION_COLUMNS:
            continue
        if col in CATEGORICAL_FEATURE_COLUMNS:
            candidates.append(col)
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
            candidates.append(col)
    return candidates


def make_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    training_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    x = frame[feature_columns].copy()
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col in x.columns:
            x[col] = x[col].astype(str)

    x = pd.get_dummies(x, columns=[col for col in CATEGORICAL_FEATURE_COLUMNS if col in x.columns])
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    if training_columns is not None:
        x = x.reindex(columns=training_columns, fill_value=0.0)
        return x, training_columns

    return x, x.columns.tolist()


def fit_binary_classifier(
    train_frame: pd.DataFrame,
    input_feature_columns: list[str],
    random_state: int = 42,
) -> FittedBinaryModel:
    x, training_columns = make_feature_matrix(train_frame, input_feature_columns)
    y = train_frame["utilized"].astype(int).to_numpy()

    if len(np.unique(y)) == 1:
        model = ConstantProbabilityModel(float(y[0]))
    else:
        model = SimpleGradientBoostingClassifier(random_state=random_state)
        model.fit(x.to_numpy(), y)

    return FittedBinaryModel(
        model=model,
        input_feature_columns=input_feature_columns,
        training_columns=training_columns,
    )


def predict_probability(fitted: FittedBinaryModel, frame: pd.DataFrame) -> np.ndarray:
    x, _ = make_feature_matrix(
        frame,
        fitted.input_feature_columns,
        training_columns=fitted.training_columns,
    )
    proba = fitted.model.predict_proba(x.to_numpy())[:, 1]
    return np.clip(proba, 0.001, 0.999)


def train_model_bundle(
    assignments: pd.DataFrame,
    train_split: str = "train",
    feature_columns: list[str] | None = None,
    include_prediction_features: bool = False,
    random_state: int = 42,
) -> tuple[ModelBundle, list[str]]:
    if feature_columns is None:
        feature_columns = infer_feature_columns(assignments, include_prediction_features=include_prediction_features)

    train = assignments.loc[(assignments["split"] == train_split) & assignments["is_available"]].copy()
    if train.empty:
        raise ValueError(f"Нет строк для train_split={train_split}.")

    if train["shown"].nunique() < 2:
        raise ValueError("Для S-learner нужны и показанные, и непоказанные строки.")

    s_learner_features = [col for col in feature_columns if col != TREATMENT_FEATURE_COLUMN]
    train = _with_treatment_value(train, treatment_value=None)
    interaction_features = _add_slearner_interactions(train, s_learner_features)
    s_learner_features = s_learner_features + [TREATMENT_FEATURE_COLUMN] + interaction_features
    outcome_model = fit_binary_classifier(train, s_learner_features, random_state=random_state)
    return ModelBundle(
        response_model=outcome_model,
        treatment_model=outcome_model,
        control_model=outcome_model,
        learner_type="s_learner",
    ), s_learner_features


def _add_slearner_interactions(frame: pd.DataFrame, base_feature_columns: list[str]) -> list[str]:
    interaction_features = []
    for col in base_feature_columns:
        if col in CATEGORICAL_FEATURE_COLUMNS:
            continue
        if col not in frame.columns:
            continue
        if not (pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col])):
            continue
        interaction_col = f"{TREATMENT_INTERACTION_PREFIX}{col}"
        frame[interaction_col] = frame[TREATMENT_FEATURE_COLUMN].astype(float) * frame[col].astype(float)
        interaction_features.append(interaction_col)
    return interaction_features


def _with_treatment_value(
    frame: pd.DataFrame,
    treatment_value: float | None,
    input_feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    if treatment_value is None:
        out[TREATMENT_FEATURE_COLUMN] = out["shown"].astype(float)
    else:
        out[TREATMENT_FEATURE_COLUMN] = float(treatment_value)
    if input_feature_columns is not None:
        base_features = [
            col.removeprefix(TREATMENT_INTERACTION_PREFIX)
            for col in input_feature_columns
            if col.startswith(TREATMENT_INTERACTION_PREFIX)
        ]
        _add_slearner_interactions(out, base_features)
    return out


def score_with_model_bundle(
    assignments: pd.DataFrame,
    bundle: ModelBundle,
    prefix: str = "",
) -> pd.DataFrame:
    scored = assignments.copy()
    if TREATMENT_FEATURE_COLUMN in bundle.treatment_model.input_feature_columns:
        treatment_frame = _with_treatment_value(scored, 1.0, bundle.treatment_model.input_feature_columns)
        control_frame = _with_treatment_value(scored, 0.0, bundle.control_model.input_feature_columns)
        scored[f"{prefix}p1_hat"] = predict_probability(bundle.treatment_model, treatment_frame)
        scored[f"{prefix}p0_hat"] = predict_probability(bundle.control_model, control_frame)
        scored[f"{prefix}p_util_hat"] = scored[f"{prefix}p1_hat"]
    else:
        scored[f"{prefix}p_util_hat"] = predict_probability(bundle.response_model, scored)
        scored[f"{prefix}p1_hat"] = predict_probability(bundle.treatment_model, scored)
        scored[f"{prefix}p0_hat"] = predict_probability(bundle.control_model, scored)
    scored[f"{prefix}uplift_hat"] = scored[f"{prefix}p1_hat"] - scored[f"{prefix}p0_hat"]
    scored[f"{prefix}response_model_score"] = scored["npv"] * scored[f"{prefix}p_util_hat"]
    scored[f"{prefix}extra_npv_model_score"] = scored["npv"] * scored[f"{prefix}uplift_hat"]
    return scored


def _make_transformer_encoder(
    events: pd.DataFrame,
    config: SyntheticConfig,
    seed: int = 42,
) -> TransformerEncoder:
    rng = np.random.default_rng(seed)
    d_model = config.latent_dim
    vocabs = build_vocabularies(events)

    embedding_tables = {}
    for col, vocab in vocabs.items():
        table = rng.normal(0.0, 0.18, size=(len(vocab) + 1, d_model))
        table[0, :] = 0.0
        embedding_tables[col] = table

    recency_embedding = rng.normal(0.0, 0.10, size=(12, d_model))
    recency_embedding[0, :] = 0.0
    position_embedding = rng.normal(0.0, 0.08, size=(config.max_seq_len, d_model))
    value_vector = rng.normal(0.0, 0.12, size=d_model)
    scale = 1.0 / np.sqrt(d_model)
    return TransformerEncoder(
        vocabs=vocabs,
        embedding_tables=embedding_tables,
        recency_embedding=recency_embedding,
        position_embedding=position_embedding,
        value_vector=value_vector,
        w_q=rng.normal(0.0, scale, size=(d_model, d_model)),
        w_k=rng.normal(0.0, scale, size=(d_model, d_model)),
        w_v=rng.normal(0.0, scale, size=(d_model, d_model)),
        w_o=rng.normal(0.0, scale, size=(d_model, d_model)),
        pool_query=rng.normal(0.0, scale, size=d_model),
        head_w=rng.normal(0.0, scale, size=4 * d_model + 2),
        head_b=0.0,
        d_model=d_model,
        is_trained=False,
        training_losses=[],
    )


def _softmax_masked(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = scores.copy()
    masked[:, ~mask] = -1e9
    masked = masked - masked.max(axis=1, keepdims=True)
    weights = np.exp(masked)
    weights[:, ~mask] = 0.0
    denom = weights.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return weights / denom


def _sequence_token_matrix(unit_seq: pd.DataFrame, encoder: TransformerEncoder) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(unit_seq), encoder.d_model), dtype=float)
    for col in EVENT_CATEGORICAL_COLUMNS:
        ids = unit_seq[f"{col}_id"].to_numpy(dtype=int)
        ids = np.clip(ids, 0, encoder.embedding_tables[col].shape[0] - 1)
        x += encoder.embedding_tables[col][ids]

    recency_ids = np.clip(unit_seq["recency_bucket"].to_numpy(dtype=int), 0, encoder.recency_embedding.shape[0] - 1)
    positions = np.clip(unit_seq["position"].to_numpy(dtype=int), 0, encoder.position_embedding.shape[0] - 1)
    x += encoder.recency_embedding[recency_ids]
    x += encoder.position_embedding[positions]
    x += unit_seq["event_value"].to_numpy(dtype=float)[:, None] * encoder.value_vector[None, :]

    non_padding = unit_seq["is_padding"].to_numpy(dtype=int) == 0
    return x, non_padding


def _self_attention_tokens(x: np.ndarray, non_padding: np.ndarray, encoder: TransformerEncoder) -> np.ndarray:
    if not non_padding.any():
        return np.zeros_like(x)

    q = x @ encoder.w_q
    k = x @ encoder.w_k
    v = x @ encoder.w_v
    scores = (q @ k.T) / np.sqrt(encoder.d_model)
    attn = _softmax_masked(scores, non_padding)
    h = (attn @ v) @ encoder.w_o
    return np.tanh(h + x)


def _attention_pool(token_h: np.ndarray, non_padding: np.ndarray, encoder: TransformerEncoder) -> tuple[np.ndarray, np.ndarray]:
    if not non_padding.any():
        return np.zeros(encoder.d_model, dtype=float), np.zeros(len(token_h), dtype=float)

    scores = (token_h @ encoder.pool_query) / np.sqrt(encoder.d_model)
    scores = np.where(non_padding, scores, -1e9)
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = np.where(non_padding, weights, 0.0)
    weights = weights / max(float(weights.sum()), 1e-12)
    return weights @ token_h, weights


def _encode_one_sequence(unit_seq: pd.DataFrame, encoder: TransformerEncoder) -> np.ndarray:
    x, non_padding = _sequence_token_matrix(unit_seq, encoder)
    token_h = _self_attention_tokens(x, non_padding, encoder)
    pooled, _ = _attention_pool(token_h, non_padding, encoder)
    return pooled


def _transformer_head_features(h: np.ndarray, row, config: SyntheticConfig) -> np.ndarray:
    cat = np.array([getattr(row, f"cat_emb_{dim}") for dim in range(config.latent_dim)], dtype=float)
    return np.concatenate(
        [
            h,
            cat,
            h * cat,
            np.abs(h - cat),
            np.array([float(row.npv) / 100.0, float(row.group_pref)], dtype=float),
        ]
    )


def _update_event_embeddings(
    encoder: TransformerEncoder,
    unit_seq: pd.DataFrame,
    token_grad: np.ndarray,
    learning_rate: float,
) -> None:
    scale = learning_rate / max(1, len(EVENT_CATEGORICAL_COLUMNS))
    for row_idx, row in enumerate(unit_seq.itertuples(index=False)):
        if int(row.is_padding) == 1:
            continue
        grad = token_grad[row_idx]
        row_dict = row._asdict()
        for col in EVENT_CATEGORICAL_COLUMNS:
            token_id = int(row_dict[f"{col}_id"])
            if token_id > 0:
                encoder.embedding_tables[col][token_id] -= scale * grad
        recency_id = int(np.clip(row.recency_bucket, 0, encoder.recency_embedding.shape[0] - 1))
        position_id = int(np.clip(row.position, 0, encoder.position_embedding.shape[0] - 1))
        encoder.recency_embedding[recency_id] -= learning_rate * 0.20 * grad
        encoder.position_embedding[position_id] -= learning_rate * 0.12 * grad
        encoder.value_vector -= learning_rate * 0.08 * float(row.event_value) * grad


def _train_transformer_encoder(
    encoder: TransformerEncoder,
    events: pd.DataFrame,
    assignments: pd.DataFrame,
    config: SyntheticConfig,
    train_split: str,
    seed: int,
    epochs: int = 7,
    learning_rate: float = 0.016,
    max_rows: int = 4000,
) -> TransformerEncoder:
    """Обучить легкий mini-transformer encoder по факту утилизации shown-категорий."""
    rng = np.random.default_rng(seed)
    train_rows = assignments.loc[
        (assignments["split"] == train_split) & assignments["shown"] & assignments["is_available"]
    ].copy()
    if train_rows.empty:
        encoder.training_losses = []
        return encoder

    if len(train_rows) > max_rows:
        train_rows = train_rows.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    train_units = decision_units_from_assignments(train_rows)
    seq = build_transformer_sequence_table(events, train_units, config, vocabs=encoder.vocabs)
    seq_by_unit = {
        key: unit_seq.sort_values("position").reset_index(drop=True)
        for key, unit_seq in seq.groupby(["user_id", "period_id"], sort=False)
    }

    losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        shuffled = train_rows.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        for row in shuffled.itertuples(index=False):
            unit_seq = seq_by_unit[(int(row.user_id), int(row.period_id))]
            x, non_padding = _sequence_token_matrix(unit_seq, encoder)
            token_h = _self_attention_tokens(x, non_padding, encoder)
            h, weights = _attention_pool(token_h, non_padding, encoder)
            features = _transformer_head_features(h, row, config)

            y = float(row.utilized)
            old_head_w = encoder.head_w.copy()
            pred = float(sigmoid(features @ old_head_w + encoder.head_b))
            grad_logit = pred - y
            epoch_loss += float(-(y * np.log(pred + 1e-12) + (1.0 - y) * np.log(1.0 - pred + 1e-12)))

            encoder.head_w -= learning_rate * grad_logit * features
            encoder.head_b -= learning_rate * grad_logit

            cat = np.array([getattr(row, f"cat_emb_{dim}") for dim in range(config.latent_dim)], dtype=float)
            grad_features = grad_logit * old_head_w
            grad_h = (
                grad_features[: config.latent_dim]
                + grad_features[2 * config.latent_dim : 3 * config.latent_dim] * cat
                + grad_features[3 * config.latent_dim : 4 * config.latent_dim] * np.sign(h - cat)
            )

            score_grad = weights * ((token_h - h) @ grad_h) / np.sqrt(encoder.d_model)
            grad_pool_query = (score_grad[:, None] * token_h).sum(axis=0)
            token_grad = weights[:, None] * grad_h[None, :]
            token_grad += score_grad[:, None] * encoder.pool_query[None, :] / np.sqrt(encoder.d_model)

            encoder.pool_query -= learning_rate * grad_pool_query
            _update_event_embeddings(encoder, unit_seq, token_grad, learning_rate)

        losses.append(epoch_loss / max(1, len(train_rows)))

    encoder.is_trained = True
    encoder.training_losses = losses
    return encoder


def encode_decision_units(
    events: pd.DataFrame,
    assignments: pd.DataFrame,
    config: SyntheticConfig,
    encoder: TransformerEncoder,
) -> pd.DataFrame:
    units = decision_units_from_assignments(assignments)
    seq = build_transformer_sequence_table(events, units, config, vocabs=encoder.vocabs)
    rows = []
    for (user_id, period_id), unit_seq in seq.groupby(["user_id", "period_id"], sort=False):
        h = _encode_one_sequence(unit_seq.sort_values("position"), encoder)
        row = {"user_id": int(user_id), "period_id": int(period_id)}
        row.update({f"tr_user_{idx}": float(value) for idx, value in enumerate(h)})
        rows.append(row)
    return pd.DataFrame(rows)


def make_transformer_pair_frame(
    assignments: pd.DataFrame,
    unit_embeddings: pd.DataFrame,
    config: SyntheticConfig,
) -> tuple[pd.DataFrame, list[str]]:
    frame = assignments.merge(unit_embeddings, on=["user_id", "period_id"], how="left")
    feature_columns = []
    for dim in range(config.latent_dim):
        user_col = f"tr_user_{dim}"
        cat_col = f"cat_emb_{dim}"
        mul_col = f"tr_mul_{dim}"
        abs_col = f"tr_absdiff_{dim}"
        frame[mul_col] = frame[user_col] * frame[cat_col]
        frame[abs_col] = (frame[user_col] - frame[cat_col]).abs()
        feature_columns.extend([user_col, cat_col, mul_col, abs_col])
    for col in TRANSFORMER_SEQUENCE_FEATURE_COLUMNS:
        if col in frame.columns:
            log_col = f"tr_{col}_log1p"
            frame[log_col] = np.log1p(frame[col].astype(float).clip(lower=0.0))
            feature_columns.append(log_col)
    feature_columns.extend(["npv", "group_pref"])
    return frame, feature_columns


def train_transformer_bundle(
    events: pd.DataFrame,
    assignments: pd.DataFrame,
    config: SyntheticConfig,
    train_split: str = "train",
    seed: int = 42,
) -> tuple[TransformerBundle, pd.DataFrame]:
    encoder = _make_transformer_encoder(events, config, seed=seed)
    encoder = _train_transformer_encoder(
        encoder,
        events=events,
        assignments=assignments,
        config=config,
        train_split=train_split,
        seed=seed + 7,
    )
    unit_embeddings = encode_decision_units(events, assignments, config, encoder)
    pair_frame, transformer_features = make_transformer_pair_frame(assignments, unit_embeddings, config)
    bundle, _ = train_model_bundle(
        pair_frame,
        train_split=train_split,
        feature_columns=transformer_features,
        random_state=seed + 10,
    )
    transformer_bundle = TransformerBundle(
        encoder=encoder,
        response_model=bundle.response_model,
        treatment_model=bundle.treatment_model,
        control_model=bundle.control_model,
        transformer_feature_columns=transformer_features,
    )
    return transformer_bundle, unit_embeddings


def score_with_transformer_bundle(
    assignments: pd.DataFrame,
    unit_embeddings: pd.DataFrame,
    config: SyntheticConfig,
    transformer_bundle: TransformerBundle,
) -> pd.DataFrame:
    pair_frame, _ = make_transformer_pair_frame(assignments, unit_embeddings, config)
    scored = assignments.copy()
    if TREATMENT_FEATURE_COLUMN in transformer_bundle.treatment_model.input_feature_columns:
        treatment_frame = _with_treatment_value(
            pair_frame,
            1.0,
            transformer_bundle.treatment_model.input_feature_columns,
        )
        control_frame = _with_treatment_value(
            pair_frame,
            0.0,
            transformer_bundle.control_model.input_feature_columns,
        )
        scored["transformer_p1_hat"] = predict_probability(transformer_bundle.treatment_model, treatment_frame)
        scored["transformer_p0_hat"] = predict_probability(transformer_bundle.control_model, control_frame)
        scored["transformer_p_util_hat"] = scored["transformer_p1_hat"]
    else:
        scored["transformer_p_util_hat"] = predict_probability(transformer_bundle.response_model, pair_frame)
        scored["transformer_p1_hat"] = predict_probability(transformer_bundle.treatment_model, pair_frame)
        scored["transformer_p0_hat"] = predict_probability(transformer_bundle.control_model, pair_frame)
    scored["transformer_uplift_hat"] = scored["transformer_p1_hat"] - scored["transformer_p0_hat"]
    return scored


def train_stacked_bundle(
    assignments_with_transformer_scores: pd.DataFrame,
    train_split: str = "train",
    seed: int = 42,
) -> tuple[ModelBundle, list[str]]:
    base_features = infer_feature_columns(assignments_with_transformer_scores, include_prediction_features=False)
    feature_columns = base_features + [col for col in TRANSFORMER_SCORE_COLUMNS if col in assignments_with_transformer_scores]
    return train_model_bundle(
        assignments_with_transformer_scores,
        train_split=train_split,
        feature_columns=feature_columns,
        include_prediction_features=True,
        random_state=seed + 20,
    )


def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _fit_platt(raw_prob: np.ndarray, y: np.ndarray, iters: int = 800, lr: float = 0.25) -> tuple[float, float]:
    """Platt scaling: подобрать (a, b) в sigmoid(a*logit(raw)+b) градиентным спуском."""
    logits = _to_logit(raw_prob)
    target = np.asarray(y, dtype=float)
    a, b = 1.0, 0.0
    n = max(1, len(target))
    for _ in range(iters):
        p = sigmoid(a * logits + b)
        grad_a = float(np.dot(p - target, logits) / n)
        grad_b = float(np.sum(p - target) / n)
        a -= lr * grad_a
        b -= lr * grad_b
    return a, b


def _apply_platt(raw_prob: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(sigmoid(a * _to_logit(raw_prob) + b), 0.001, 0.999)


def calibrate_stacked_scores(
    scored: pd.DataFrame,
    calib_splits: tuple[str, ...] = ("calib", "val", "train"),
) -> pd.DataFrame:
    """Откалибровать stacked-прогнозы Platt scaling по отложенному калибровочному срезу.

    Это настоящая per-arm калибровка вероятностей, а не ручной ансамбль: для головы
    показа (`p1`) параметры подбираются на shown-строках калибровочного среза, для
    головы контроля (`p0`) — на not-shown строках. Калибруется наблюдаемая частота
    утилизации, поэтому stacking остается самостоятельной моделью на расширенных
    признаках, а не сглаживается к трансформеру. Если отдельного `calib` среза нет,
    используется первый доступный из `calib_splits`.
    """
    out = scored.copy()
    calib = None
    for split in calib_splits:
        candidate = out.loc[out["split"] == split]
        if not candidate.empty:
            calib = candidate
            break
    if calib is None:
        return out

    arms = [
        ("stacked_p1_hat", calib.loc[calib["shown"].astype(bool) & calib["is_available"]]),
        ("stacked_p0_hat", calib.loc[~calib["shown"].astype(bool) & calib["is_available"]]),
    ]
    for col, arm_frame in arms:
        if col not in out.columns or arm_frame.empty or arm_frame["utilized"].nunique() < 2:
            continue
        a, b = _fit_platt(arm_frame[col].to_numpy(), arm_frame["utilized"].to_numpy())
        out[col] = _apply_platt(out[col].to_numpy(), a, b)

    if {"stacked_p_util_hat", "stacked_p1_hat"}.issubset(out.columns):
        out["stacked_p_util_hat"] = out["stacked_p1_hat"]
    if {"stacked_p1_hat", "stacked_p0_hat"}.issubset(out.columns):
        out["stacked_uplift_hat"] = out["stacked_p1_hat"] - out["stacked_p0_hat"]
    return out
