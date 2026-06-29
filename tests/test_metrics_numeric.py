"""Численная проверка самописных метрик.

Проект намеренно не зависит от scikit-learn (только numpy/pandas), поэтому
эталонные значения зашиты константами. Они один раз сверены со scikit-learn
1.4.2: на этом наборе ROC AUC, Average Precision и Brier совпадают до 1e-10.
Дополнительно проверяются аналитические граничные случаи, где значение метрики
известно точно.
"""

import numpy as np
import pytest

from utils.metrics import average_precision_score, brier_score, roc_auc_score


# Фиксированный набор без совпадающих скоров.
Y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
S = np.array(
    [0.31, 0.62, 0.55, 0.40, 0.48, 0.66, 0.21, 0.72, 0.39, 0.58,
     0.77, 0.44, 0.51, 0.69, 0.33, 0.60, 0.45, 0.29, 0.81, 0.50,
     0.37, 0.64, 0.47, 0.53, 0.42]
)

# Эталоны из scikit-learn 1.4.2.
SK_AUC = 0.7756410256
SK_AP = 0.8052233023
SK_BRIER = 0.1987600000


def test_roc_auc_matches_sklearn_reference() -> None:
    assert roc_auc_score(Y, S) == pytest.approx(SK_AUC, abs=1e-9)


def test_average_precision_matches_sklearn_reference() -> None:
    assert average_precision_score(Y, S) == pytest.approx(SK_AP, abs=1e-9)


def test_brier_matches_sklearn_reference() -> None:
    assert brier_score(Y, S) == pytest.approx(SK_BRIER, abs=1e-9)


def test_roc_auc_with_ties() -> None:
    # На совпадающих скорах ROC AUC использует усреднение рангов; эталон sklearn = 0.64.
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.9, 0.1, 0.2, 0.9, 0.2, 0.5, 0.7])
    assert roc_auc_score(y, s) == pytest.approx(0.64, abs=1e-9)


def test_roc_auc_perfect_and_reversed() -> None:
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    reversed_scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc_score(y, perfect) == pytest.approx(1.0)
    assert roc_auc_score(y, reversed_scores) == pytest.approx(0.0)


def test_average_precision_perfect_ranking() -> None:
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert average_precision_score(y, perfect) == pytest.approx(1.0)


def test_brier_constant_predictions() -> None:
    y = np.array([1, 1, 0, 0])
    half = np.full(4, 0.5)
    # Каждая ошибка (0.5)^2 = 0.25.
    assert brier_score(y, half) == pytest.approx(0.25)


def test_metrics_degenerate_single_class_return_nan() -> None:
    y = np.array([1, 1, 1])
    s = np.array([0.2, 0.6, 0.9])
    assert np.isnan(roc_auc_score(y, s))
    assert np.isnan(average_precision_score(np.array([0, 0, 0]), s))
