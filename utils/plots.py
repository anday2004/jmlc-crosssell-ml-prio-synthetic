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
    """Сохранить bar chart по главной метрике (true extra NPV, инкремент) и SNIPS.

    Главная ось — добавленная ценность (true extra NPV): именно её максимизирует
    extra NPV-ранжирование. SNIPS (наблюдаемая утилизация, ценность отклика)
    показан точками как вторичная справочная метрика на верхней оси.
    """
    required = {"policy_name", "snips_value", "true_extra_npv_value"}
    missing = required - set(evaluation.columns)
    if missing:
        raise ValueError(f"В evaluation не хватает колонок: {sorted(missing)}")

    plot_frame = evaluation.sort_values("true_extra_npv_value", ascending=True).copy()
    labels = plot_frame["policy_name"].tolist()
    y_pos = list(range(len(plot_frame)))

    plt = _plt()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(plot_frame))))
    bars = ax.barh(y_pos, plot_frame["true_extra_npv_value"], color="#8c1822", alpha=0.92,
                   label="true extra NPV (инкремент, главная)")
    ax.set_yticks(y_pos, labels=labels)
    ax.set_xlabel("Добавленная ценность — true extra NPV (инкремент)")
    ax.set_title("Сравнение стратегий по добавленной ценности")
    ax.grid(axis="x", alpha=0.25)

    ax2 = ax.twiny()
    pts = ax2.scatter(plot_frame["snips_value"], y_pos, color="#e0902c", zorder=3,
                      edgecolors="#5a3000", linewidths=0.5, s=46,
                      label="SNIPS (отклик, вторичная)")
    ax2.set_xlabel("SNIPS — наблюдаемая утилизация (отклик)")

    ax.legend(handles=[bars, pts], loc="lower right", fontsize=9)
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

    palette = {"p_util_hat": "#8c1822", "transformer_p_util_hat": "#e0902c", "stacked_p_util_hat": "#5a6b7a"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in score_columns:
        ax.hist(scored[col], bins=30, alpha=0.5, density=True, label=col, color=palette.get(col))
    ax.set_xlabel("Прогноз вероятности утилизации")
    ax.set_ylabel("Плотность")
    ax.set_title("Распределения модельных скорингов")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
