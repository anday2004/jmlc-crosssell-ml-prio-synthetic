"""Быстрые smoke-тесты синтетического пайплайна."""


def test_skeleton_imports() -> None:
    import utils.data  # noqa: F401
    import utils.evaluation  # noqa: F401
    import utils.metrics  # noqa: F401
    import utils.models  # noqa: F401
    import utils.plots  # noqa: F401
    import utils.policies  # noqa: F401


def test_synthetic_generator_contract() -> None:
    from utils.data import SyntheticConfig, generate_synthetic_data, validate_synthetic_dataset

    config = SyntheticConfig(
        seed=11,
        n_users=24,
        n_categories=12,
        n_periods=5,
        min_history_periods=2,
        n_slots=3,
        max_seq_len=12,
    )
    dataset = generate_synthetic_data(config)

    assert {"user_id", "activity_level", "recommendation_sensitivity"}.issubset(dataset.users.columns)
    assert {"category_id", "group_id", "npv"}.issubset(dataset.categories.columns)
    assert {"event_family", "event_type", "event_time"}.issubset(dataset.events.columns)
    assert {"shown", "show_propensity", "p0_true", "p1_true", "extra_npv_true"}.issubset(
        dataset.assignments.columns
    )

    checks = validate_synthetic_dataset(dataset, config)
    assert checks["fixed_slots"]
    assert checks["positive_propensity_for_shown"]
    assert checks["no_future_events_in_history"]
    assert checks["fixed_sequence_length"]
    assert checks["non_negative_recency_bucket"]
    assert checks["oracle_has_signal"]


def test_greedy_policies_use_unique_groups() -> None:
    from utils.data import SyntheticConfig, generate_synthetic_data
    from utils.policies import make_policy_family, policy_slot_counts

    config = SyntheticConfig(
        seed=13,
        n_users=20,
        n_categories=12,
        n_periods=5,
        min_history_periods=2,
        n_slots=3,
    )
    dataset = generate_synthetic_data(config)
    policies = make_policy_family(dataset.assignments, n_slots=config.n_slots, seed=config.seed)
    counts = policy_slot_counts(policies)

    assert (counts["n_slots"] == config.n_slots).all()
    assert (counts["n_groups"] == config.n_slots).all()


def test_policy_evaluation_returns_core_metrics() -> None:
    from utils.data import SyntheticConfig, generate_synthetic_data
    from utils.evaluation import evaluate_policies
    from utils.policies import make_policy_family

    config = SyntheticConfig(
        seed=17,
        n_users=24,
        n_categories=12,
        n_periods=5,
        min_history_periods=2,
        n_slots=3,
    )
    dataset = generate_synthetic_data(config)
    policies = make_policy_family(dataset.assignments, n_slots=config.n_slots, seed=config.seed)
    result = evaluate_policies(
        dataset.assignments,
        policies,
        n_slots=config.n_slots,
        n_bootstrap=20,
        seed=config.seed,
    )

    assert {"ips_value", "snips_value", "dr_value", "ess", "matched_rows", "true_extra_npv_value"}.issubset(
        result.columns
    )
    assert set(result["policy_name"]) == {"random", "npv", "response_true", "oracle_extra_npv"}
    assert (result["matched_rows"] > 0).all()
    assert (result["ess"] > 0).all()


def test_model_transformer_and_stacking_pipeline() -> None:
    from utils.data import SyntheticConfig, generate_synthetic_data
    from utils.evaluation import evaluate_policies
    from utils.metrics import evaluate_model_quality, evaluate_uplift_alignment
    from utils.models import (
        score_with_model_bundle,
        score_with_transformer_bundle,
        train_model_bundle,
        train_stacked_bundle,
        train_transformer_bundle,
    )
    from utils.policies import make_model_policy_family, policy_slot_counts

    config = SyntheticConfig(
        seed=23,
        n_users=24,
        n_categories=12,
        n_periods=5,
        min_history_periods=2,
        n_slots=3,
        max_seq_len=16,
    )
    dataset = generate_synthetic_data(config)

    tabular_bundle, tabular_features = train_model_bundle(dataset.assignments, random_state=config.seed)
    scored = score_with_model_bundle(dataset.assignments, tabular_bundle)

    transformer_bundle, unit_embeddings = train_transformer_bundle(
        dataset.events,
        scored,
        config,
        seed=config.seed,
    )
    scored = score_with_transformer_bundle(scored, unit_embeddings, config, transformer_bundle)

    stacked_bundle, stacked_features = train_stacked_bundle(scored, seed=config.seed)
    scored = score_with_model_bundle(scored, stacked_bundle, prefix="stacked_")

    expected_score_columns = {
        "p_util_hat",
        "p1_hat",
        "p0_hat",
        "transformer_p_util_hat",
        "transformer_p1_hat",
        "transformer_p0_hat",
        "stacked_p_util_hat",
        "stacked_p1_hat",
        "stacked_p0_hat",
    }
    assert expected_score_columns.issubset(scored.columns)
    assert tabular_features
    assert transformer_bundle.transformer_feature_columns
    assert transformer_bundle.encoder.is_trained
    assert transformer_bundle.encoder.training_losses
    assert stacked_features
    assert not unit_embeddings.empty

    policies = make_model_policy_family(scored, n_slots=config.n_slots)
    counts = policy_slot_counts(policies)
    assert set(policies["policy_name"]) == {
        "response_model",
        "extra_npv_model",
        "response_transformer",
        "extra_npv_transformer",
        "response_stacked",
        "extra_npv_stacked",
    }
    assert (counts["n_slots"] == config.n_slots).all()
    assert (counts["n_groups"] == config.n_slots).all()

    result = evaluate_policies(
        scored,
        policies,
        n_slots=config.n_slots,
        n_bootstrap=10,
        seed=config.seed,
    )
    model_quality = evaluate_model_quality(scored)
    uplift_alignment = evaluate_uplift_alignment(scored)

    assert {"ips_value", "snips_value", "dr_value", "true_extra_npv_value"}.issubset(result.columns)
    assert result["policy_name"].nunique() == 6
    assert {"roc_auc", "average_precision", "brier"}.issubset(model_quality.columns)
    assert {"extra_npv_corr", "top_decile_lift"}.issubset(uplift_alignment.columns)
    assert not model_quality.empty
    assert not uplift_alignment.empty


def test_default_seed_policy_sanity_on_test_split() -> None:
    from run_experiment import run_pipeline
    from utils.data import SyntheticConfig

    config = SyntheticConfig(seed=42, n_users=80, n_categories=12, n_periods=6)
    result = run_pipeline(config, n_bootstrap=5)
    evaluation = result["evaluation"]

    random_true = float(evaluation.loc[evaluation["policy_name"] == "random", "true_extra_npv_value"].iloc[0])
    oracle_true = float(
        evaluation.loc[evaluation["policy_name"] == "oracle_extra_npv", "true_extra_npv_value"].iloc[0]
    )
    model_true = evaluation.loc[
        evaluation["policy_name"].isin(
            [
                "response_model",
                "extra_npv_model",
                "response_transformer",
                "extra_npv_transformer",
                "response_stacked",
                "extra_npv_stacked",
            ]
        ),
        "true_extra_npv_value",
    ]

    assert oracle_true == evaluation["true_extra_npv_value"].max()
    assert model_true.max() > random_true * 1.35
    assert "evaluation_all" in result
    assert result["evaluation"]["policy_name"].nunique() == result["evaluation_all"]["policy_name"].nunique()
