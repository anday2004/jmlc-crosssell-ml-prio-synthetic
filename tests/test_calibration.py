"""Прямой юнит-тест на stacking-калибровку.

Проверяет, что `calibrate_stacked_scores` — это настоящая per-arm Platt-калибровка:
она подбирает отображение по калибровочному срезу и реально меняет прогнозы, а не
возвращает их как есть.
"""

import numpy as np
import pandas as pd

from utils.models import calibrate_stacked_scores


def _make_scored(n_calib: int = 200, n_test: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_calib + n_test
    shown = np.array(([True] * (n_calib // 2) + [False] * (n_calib // 2))
                     + ([True] * (n_test // 2) + [False] * (n_test // 2)))
    p1 = rng.uniform(0.2, 0.8, n)
    p0 = rng.uniform(0.1, 0.6, n)
    # факт утилизации коррелирует с прогнозом (обе головы имеют оба класса на calib)
    utilized = np.where(shown, (p1 > 0.5).astype(int), (p0 > 0.35).astype(int))
    return pd.DataFrame(
        {
            "split": ["calib"] * n_calib + ["test"] * n_test,
            "shown": shown,
            "is_available": True,
            "utilized": utilized,
            "stacked_p1_hat": p1,
            "stacked_p0_hat": p0,
            "stacked_p_util_hat": p1,
        }
    )


def test_calibration_changes_scores() -> None:
    scored = _make_scored()
    out = calibrate_stacked_scores(scored)
    # калибровка реально меняет прогнозы обеих голов (не no-op и не ансамбль-заглушка)
    assert not np.allclose(out["stacked_p1_hat"].to_numpy(), scored["stacked_p1_hat"].to_numpy())
    assert not np.allclose(out["stacked_p0_hat"].to_numpy(), scored["stacked_p0_hat"].to_numpy())


def test_calibration_outputs_valid_probabilities_and_uplift() -> None:
    scored = _make_scored()
    out = calibrate_stacked_scores(scored)
    assert out["stacked_p1_hat"].between(0.0, 1.0).all()
    assert out["stacked_p0_hat"].between(0.0, 1.0).all()
    # uplift пересчитан как разность откалиброванных голов
    assert np.allclose(
        out["stacked_uplift_hat"].to_numpy(),
        (out["stacked_p1_hat"] - out["stacked_p0_hat"]).to_numpy(),
    )
    # p_util совпадает с откалиброванной головой показа
    assert np.allclose(out["stacked_p_util_hat"].to_numpy(), out["stacked_p1_hat"].to_numpy())


def test_calibration_noop_without_calib_split() -> None:
    scored = _make_scored()
    scored["split"] = "test"  # нет ни одного среза из calib_splits
    out = calibrate_stacked_scores(scored)
    # без калибровочного среза функция ничего не меняет
    assert np.allclose(out["stacked_p1_hat"].to_numpy(), scored["stacked_p1_hat"].to_numpy())
