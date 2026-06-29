"""Графики для диагностики и сравнения стратегий.

Модуль импортирует matplotlib только внутри функций, чтобы основной пайплайн
мог запускаться даже в минимальном окружении.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_policy_comparison(evaluation: pd.DataFrame, output_path: str | Path) -> Path:
    """Сохранить bar chart по IPS/SNIPS и истинному synthetic extra NPV."""
    required = {"policy_name", "ips_value", "snips_value", "true_extra_npv_value"}
    missing = required - set(evaluation.columns)
    if missing:
        raise ValueError(f"В evaluation не хватает колонок: {sorted(missing)}")

    plot_frame = evaluation.sort_values("snips_value", ascending=True).copy()
    labels = plot_frame["policy_name"].tolist()
    y_pos = range(len(plot_frame))

    plt = _plt()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(plot_frame))))
    ax.barh(y_pos, plot_frame["snips_value"], label="SNIPS", color="#2f6f9f", alpha=0.86)
    ax.scatter(plot_frame["ips_value"], y_pos, label="IPS", color="#d95f02", zorder=3)
    ax.set_yticks(list(y_pos), labels=labels)
    ax.set_xlabel("Оценка ценности стратегии")
    ax.set_title("Сравнение стратегий")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_score_distributions(scored: pd.DataFrame, output_path: str | Path) -> Path:
    """Сохранить распределения прогнозов табличной, transformer и stacking моделей."""
    score_columns = [
        col
        for col in ["p_util_hat", "transformer_p_util_hat", "stacked_p_util_hat"]
        if col in scored.columns
    ]
    if not score_columns:
        raise ValueError("В scored нет модельных колонок для графика.")

    plt = _plt()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in score_columns:
        ax.hist(scored[col], bins=30, alpha=0.45, density=True, label=col)
    ax.set_xlabel("Прогноз вероятности утилизации")
    ax.set_ylabel("Плотность")
    ax.set_title("Распределения модельных скорингов")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
