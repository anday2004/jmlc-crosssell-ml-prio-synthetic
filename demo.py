"""Короткий конкурсный demo-запуск проекта."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_experiment import run_pipeline, save_artifacts
from utils.data import SyntheticConfig
from utils.plots import plot_policy_comparison, plot_score_distributions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запустить компактный demo CrossSell extra NPV.")
    parser.add_argument("--users", type=int, default=80, help="Число синтетических пользователей.")
    parser.add_argument("--categories", type=int, default=12, help="Число синтетических категорий.")
    parser.add_argument("--periods", type=int, default=6, help="Число временных периодов.")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно.")
    parser.add_argument("--bootstrap", type=int, default=80, help="Число bootstrap-итераций.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/demo", help="Куда сохранить артефакты.")
    parser.add_argument("--plots", action="store_true", help="Попробовать сохранить PNG-графики.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticConfig(
        seed=args.seed,
        n_users=args.users,
        n_categories=args.categories,
        n_periods=args.periods,
    )

    print("Запускаю синтетический CrossSell extra NPV demo...")
    results = run_pipeline(config, n_bootstrap=args.bootstrap)
    out_dir = Path(args.artifacts_dir)
    written = save_artifacts(
        out_dir,
        dataset=results["dataset"],
        scored=results["scored"],
        policy_assignments=results["policy_assignments"],
        policy_counts=results["policy_counts"],
        evaluation=results["evaluation"],
        evaluation_all=results["evaluation_all"],
        model_quality=results["model_quality"],
        model_quality_all=results["model_quality_all"],
        uplift_alignment=results["uplift_alignment"],
        uplift_alignment_all=results["uplift_alignment_all"],
        checks=results["checks"],
        sequence_sample=results["sequence_sample"],
    )

    print("\nПроверки данных:")
    for name, value in results["checks"].items():
        print(f"  {name}: {value}")

    print("\nТоп стратегий по category-level SNIPS на test split:")
    cols = [
        "policy_name",
        "snips_value",
        "ips_value",
        "set_snips_value",
        "set_ips_value",
        "dr_value",
        "true_extra_npv_value",
        "ess",
        "set_ess",
    ]
    print(
        results["evaluation"][cols]
        .sort_values("snips_value", ascending=False)
        .round(4)
        .to_string(index=False)
    )

    print("\nКачество модельных голов:")
    metric_cols = ["model_name", "subset", "roc_auc", "average_precision", "brier"]
    print(results["model_quality"][metric_cols].round(4).to_string(index=False))

    print("\nUplift alignment:")
    align_cols = ["model_name", "extra_npv_corr", "top_decile_lift"]
    print(results["uplift_alignment"][align_cols].round(4).to_string(index=False))

    if args.plots:
        try:
            plot_policy_comparison(results["evaluation"], out_dir / "policy_comparison.png")
            plot_score_distributions(results["scored"], out_dir / "score_distributions.png")
            print("\nPNG-графики сохранены в папку артефактов.")
        except ModuleNotFoundError as exc:
            print(f"\nPNG-графики не построены: не установлена зависимость {exc.name}.")

    print("\nАртефакты сохранены:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
