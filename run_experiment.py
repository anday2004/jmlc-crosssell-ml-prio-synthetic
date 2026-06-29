"""Точка входа для синтетического эксперимента CrossSell extra NPV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.data import (
    SyntheticConfig,
    build_transformer_sequence_table,
    decision_units_from_assignments,
    generate_synthetic_data,
    validate_synthetic_dataset,
)
from utils.evaluation import evaluate_policies
from utils.metrics import evaluate_model_quality, evaluate_uplift_alignment
from utils.models import (
    calibrate_stacked_scores,
    score_with_model_bundle,
    score_with_transformer_bundle,
    train_model_bundle,
    train_stacked_bundle,
    train_transformer_bundle,
)
from utils.plots import plot_policy_comparison, plot_score_distributions
from utils.policies import make_model_policy_family, make_policy_family, policy_slot_counts


MODEL_COLS = [
    "p_util_hat",
    "p1_hat",
    "p0_hat",
    "transformer_p_util_hat",
    "transformer_p1_hat",
    "transformer_p0_hat",
    "stacked_p_util_hat",
    "stacked_p1_hat",
    "stacked_p0_hat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сгенерировать синтетический CrossSell-датасет.")
    parser.add_argument("--users", type=int, default=300, help="Число синтетических пользователей.")
    parser.add_argument("--categories", type=int, default=18, help="Число CrossSell-категорий.")
    parser.add_argument("--periods", type=int, default=8, help="Число временных периодов.")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно.")
    parser.add_argument("--bootstrap", type=int, default=100, help="Число bootstrap-итераций для IPS/SNIPS.")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Папка для сохранения CSV-артефактов.")
    parser.add_argument("--plots-dir", type=str, default=None, help="Папка для сохранения диагностических PNG.")
    return parser.parse_args()


def save_artifacts(
    artifacts_dir: str | Path,
    dataset,
    scored: pd.DataFrame,
    policy_assignments: pd.DataFrame,
    policy_counts: pd.DataFrame,
    evaluation: pd.DataFrame,
    evaluation_all: pd.DataFrame,
    model_quality: pd.DataFrame,
    model_quality_all: pd.DataFrame,
    uplift_alignment: pd.DataFrame,
    uplift_alignment_all: pd.DataFrame,
    checks: dict[str, bool],
    sequence_sample: pd.DataFrame,
) -> list[Path]:
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_summary = scored[MODEL_COLS].describe().round(6).reset_index().rename(columns={"index": "metric"})
    availability_summary = (
        scored.groupby(["split", "group_id"], as_index=False)
        .agg(
            rows=("category_id", "size"),
            available_share=("is_available", "mean"),
            shown_share=("shown", "mean"),
            utilization_rate=("utilized", "mean"),
            mean_extra_npv_true=("extra_npv_true", "mean"),
        )
        .sort_values(["split", "group_id"])
    )

    files = {
        "evaluation.csv": evaluation,
        "evaluation_test.csv": evaluation,
        "evaluation_all.csv": evaluation_all,
        "model_quality.csv": model_quality,
        "model_quality_test.csv": model_quality,
        "model_quality_all.csv": model_quality_all,
        "uplift_alignment.csv": uplift_alignment,
        "uplift_alignment_test.csv": uplift_alignment,
        "uplift_alignment_all.csv": uplift_alignment_all,
        "policy_assignments.csv": policy_assignments,
        "policy_slot_counts.csv": policy_counts,
        "checks.csv": pd.DataFrame([checks]),
        "model_score_summary.csv": score_summary,
        "availability_summary.csv": availability_summary,
        "users_sample.csv": dataset.users.head(200),
        "categories.csv": dataset.categories,
        "events_sample.csv": dataset.events.head(2000),
        "assignments_scored_sample.csv": scored.head(2000),
        "transformer_sequence_sample.csv": sequence_sample,
    }

    written = []
    for file_name, frame in files.items():
        path = out_dir / file_name
        frame.to_csv(path, index=False)
        written.append(path)

    top_policies = evaluation.sort_values("snips_value", ascending=False).head(5)
    readme_lines = [
        "# Артефакты запуска",
        "",
        "## Размеры таблиц",
        "",
        f"- `users`: {dataset.users.shape[0]} строк, {dataset.users.shape[1]} колонок",
        f"- `categories`: {dataset.categories.shape[0]} строк, {dataset.categories.shape[1]} колонок",
        f"- `events`: {dataset.events.shape[0]} строк, {dataset.events.shape[1]} колонок",
        f"- `assignments`: {dataset.assignments.shape[0]} строк, {dataset.assignments.shape[1]} колонок",
        "",
        "## Проверки",
        "",
        *[f"- `{name}`: {value}" for name, value in checks.items()],
        "",
        "## Топ стратегий по category-level SNIPS на test split",
        "",
        "| policy_name | snips_value | ips_value | set_snips_value | set_ips_value | true_extra_npv_value |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_policies.itertuples(index=False):
        readme_lines.append(
            f"| `{row.policy_name}` | {row.snips_value:.4f} | "
            f"{row.ips_value:.4f} | {row.set_snips_value:.4f} | "
            f"{row.set_ips_value:.4f} | {row.true_extra_npv_value:.4f} |"
        )
    if not model_quality.empty:
        readme_lines.extend(
            [
                "",
                "## Качество модельных голов",
                "",
                "| model_name | subset | roc_auc | average_precision | brier |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in model_quality.sort_values(["head", "model_name"]).itertuples(index=False):
            readme_lines.append(
                f"| `{row.model_name}` | `{row.subset}` | {row.roc_auc:.4f} | "
                f"{row.average_precision:.4f} | {row.brier:.4f} |"
            )
    if not uplift_alignment.empty:
        readme_lines.extend(
            [
                "",
                "## Uplift Alignment",
                "",
                "| model_name | extra_npv_corr | top_decile_lift |",
                "|---|---:|---:|",
            ]
        )
        for row in uplift_alignment.sort_values("extra_npv_corr", ascending=False).itertuples(index=False):
            readme_lines.append(f"| `{row.model_name}` | {row.extra_npv_corr:.4f} | {row.top_decile_lift:.4f} |")
    readme_lines.extend(
        [
            "",
            "Файлы в этой папке являются синтетическими и не содержат рабочих данных.",
        ]
    )
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    written.append(readme_path)
    return written


def run_pipeline(config: SyntheticConfig, n_bootstrap: int = 100) -> dict:
    """Запустить полный синтетический эксперимент и вернуть все ключевые таблицы."""
    dataset = generate_synthetic_data(config)
    decision_units = decision_units_from_assignments(dataset.assignments)
    sequence_sample = build_transformer_sequence_table(
        dataset.events,
        decision_units.head(100),
        config,
    )
    checks = validate_synthetic_dataset(dataset, config)

    baseline_policy_assignments = make_policy_family(dataset.assignments, n_slots=config.n_slots, seed=config.seed)

    tabular_bundle, _ = train_model_bundle(dataset.assignments, random_state=config.seed)
    scored = score_with_model_bundle(dataset.assignments, tabular_bundle)

    transformer_bundle, unit_embeddings = train_transformer_bundle(
        dataset.events,
        scored,
        config,
        seed=config.seed,
    )
    scored = score_with_transformer_bundle(scored, unit_embeddings, config, transformer_bundle)

    stacked_bundle, _ = train_stacked_bundle(scored, seed=config.seed)
    scored = score_with_model_bundle(scored, stacked_bundle, prefix="stacked_")
    scored = calibrate_stacked_scores(scored)

    model_policy_assignments = make_model_policy_family(scored, n_slots=config.n_slots)
    policy_assignments = (
        baseline_policy_assignments
        if model_policy_assignments.empty
        else pd.concat([baseline_policy_assignments, model_policy_assignments], ignore_index=True)
    )
    policy_counts = policy_slot_counts(policy_assignments)
    evaluation_all = evaluate_policies(
        scored,
        policy_assignments,
        n_slots=config.n_slots,
        split=None,
        n_bootstrap=n_bootstrap,
        seed=config.seed,
    )
    evaluation = evaluate_policies(
        scored,
        policy_assignments,
        n_slots=config.n_slots,
        split="test",
        n_bootstrap=n_bootstrap,
        seed=config.seed,
    )
    model_quality_all = evaluate_model_quality(scored)
    model_quality = evaluate_model_quality(scored, split="test")
    uplift_alignment_all = evaluate_uplift_alignment(scored)
    uplift_alignment = evaluate_uplift_alignment(scored, split="test")

    return {
        "dataset": dataset,
        "decision_units": decision_units,
        "sequence_sample": sequence_sample,
        "checks": checks,
        "scored": scored,
        "policy_assignments": policy_assignments,
        "policy_counts": policy_counts,
        "evaluation": evaluation,
        "evaluation_all": evaluation_all,
        "model_quality": model_quality,
        "model_quality_all": model_quality_all,
        "uplift_alignment": uplift_alignment,
        "uplift_alignment_all": uplift_alignment_all,
    }


def main() -> None:
    args = parse_args()
    config = SyntheticConfig(
        seed=args.seed,
        n_users=args.users,
        n_categories=args.categories,
        n_periods=args.periods,
    )
    results = run_pipeline(config, n_bootstrap=args.bootstrap)

    dataset = results["dataset"]
    sequence_sample = results["sequence_sample"]
    checks = results["checks"]
    scored = results["scored"]
    policy_assignments = results["policy_assignments"]
    policy_counts = results["policy_counts"]
    evaluation = results["evaluation"]
    evaluation_all = results["evaluation_all"]
    model_quality = results["model_quality"]
    model_quality_all = results["model_quality_all"]
    uplift_alignment = results["uplift_alignment"]
    uplift_alignment_all = results["uplift_alignment_all"]

    print("Синтетический датасет сгенерирован.")
    print(f"users:       {dataset.users.shape}")
    print(f"categories:  {dataset.categories.shape}")
    print(f"events:      {dataset.events.shape}")
    print(f"assignments: {dataset.assignments.shape}")
    print(f"seq sample:  {sequence_sample.shape}")
    print("Проверки:")
    for name, value in checks.items():
        print(f"  {name}: {value}")

    summary = dataset.assignments[["shown", "is_available", "utilized", "p0_true", "p1_true", "extra_npv_true"]]
    print("\nКраткая статистика assignments:")
    print(summary.describe().round(4))

    print("\nСтратегии построены:")
    print(policy_assignments.groupby("policy_name")["category_id"].size().to_string())
    print("\nПроверка слотов и уникальных group_id:")
    print(policy_counts.groupby("policy_name")[["n_slots", "n_groups"]].min().to_string())

    print("\nОффлайн-оценка стратегий на test split:")
    cols = [
        "policy_name",
        "ips_value",
        "snips_value",
        "set_ips_value",
        "set_snips_value",
        "dr_value",
        "set_dr_value",
        "ess",
        "set_ess",
        "matched_rows",
        "matched_share",
        "set_matched_units",
        "set_matched_share",
        "true_shown_value",
        "true_extra_npv_value",
    ]
    print(evaluation[cols].round(4).to_string(index=False))

    print("\nОффлайн-оценка стратегий на всех split:")
    print(evaluation_all[cols].round(4).to_string(index=False))

    print("\nМодельные колонки добавлены:")
    print(scored[MODEL_COLS].describe().round(4).to_string())

    print("\nКачество модельных голов:")
    print(model_quality.round(4).to_string(index=False))

    print("\nСогласованность uplift с синтетическим эффектом:")
    print(uplift_alignment.round(4).to_string(index=False))

    if args.artifacts_dir:
        written = save_artifacts(
            args.artifacts_dir,
            dataset=dataset,
            scored=scored,
            policy_assignments=policy_assignments,
            policy_counts=policy_counts,
            evaluation=evaluation,
            evaluation_all=evaluation_all,
            model_quality=model_quality,
            model_quality_all=model_quality_all,
            uplift_alignment=uplift_alignment,
            uplift_alignment_all=uplift_alignment_all,
            checks=checks,
            sequence_sample=sequence_sample,
        )
        print("\nCSV-артефакты сохранены:")
        for path in written:
            print(f"  {path}")

    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
        try:
            policy_plot = plot_policy_comparison(evaluation, plots_dir / "policy_comparison.png")
            score_plot = plot_score_distributions(scored, plots_dir / "score_distributions.png")
            print("\nГрафики сохранены:")
            print(f"  {policy_plot}")
            print(f"  {score_plot}")
        except ModuleNotFoundError as exc:
            print(f"\nГрафики не построены: не установлена зависимость {exc.name}.")


if __name__ == "__main__":
    main()
