"""Иллюстративный Airflow DAG для синтетического CrossSell extra NPV пайплайна.

Это НЕ продакшн-DAG. В закрытом рабочем контуре оркестрация (сбор данных,
скоринг всех пар пользователь--категория, применение продуктовых правил)
живет под NDA в GitLab + Airflow. Здесь показана та же декомпозиция на шаги,
но поверх публичной синтетики: каждый таск вызывает те же функции из ``utils/``,
что и ``run_experiment.py``, и обменивается данными через директорию запуска.

Запуск шагов между собой идет через диск (parquet/csv в RUN_DIR), а не через
XCom, потому что между шагами передаются полные таблицы — это стандартная
практика для табличных ML-пайплайнов в Airflow.

Чтобы запарсить/запустить файл, нужен Apache Airflow (см. requirements-airflow.txt).
Сами шаговые функции написаны как обычные Python-функции и не зависят от Airflow,
поэтому их логику можно вызывать и без планировщика.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from utils.data import SyntheticConfig, generate_synthetic_data
from utils.evaluation import evaluate_policies
from utils.models import (
    calibrate_stacked_scores,
    score_with_model_bundle,
    score_with_transformer_bundle,
    train_model_bundle,
    train_stacked_bundle,
    train_transformer_bundle,
)
from utils.policies import make_model_policy_family, make_policy_family


RUN_DIR = Path(os.environ.get("CROSSSELL_RUN_DIR", "artifacts/airflow_run"))
CONFIG = SyntheticConfig(seed=42, n_users=80, n_categories=12, n_periods=6)


# --- Шаговые функции (не зависят от Airflow, поэтому переиспользуемы и тестируемы) ---


def step_generate_data() -> None:
    """Сгенерировать синтетические таблицы и сохранить их в RUN_DIR."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_data(CONFIG)
    dataset.users.to_parquet(RUN_DIR / "users.parquet")
    dataset.categories.to_parquet(RUN_DIR / "categories.parquet")
    dataset.events.to_parquet(RUN_DIR / "events.parquet")
    dataset.assignments.to_parquet(RUN_DIR / "assignments.parquet")


def step_score_pairs() -> None:
    """Обучить бустинг, mini-transformer и stacking; проскорить все пары."""
    assignments = pd.read_parquet(RUN_DIR / "assignments.parquet")
    events = pd.read_parquet(RUN_DIR / "events.parquet")

    tabular_bundle, _ = train_model_bundle(assignments, random_state=CONFIG.seed)
    scored = score_with_model_bundle(assignments, tabular_bundle)

    transformer_bundle, unit_embeddings = train_transformer_bundle(events, scored, CONFIG, seed=CONFIG.seed)
    scored = score_with_transformer_bundle(scored, unit_embeddings, CONFIG, transformer_bundle)

    stacked_bundle, _ = train_stacked_bundle(scored, seed=CONFIG.seed)
    scored = score_with_model_bundle(scored, stacked_bundle, prefix="stacked_")
    scored = calibrate_stacked_scores(scored)

    scored.to_parquet(RUN_DIR / "scored.parquet")


def step_build_policies() -> None:
    """Собрать базовые и модельные стратегии (greedy top-3, уникальные group_id)."""
    assignments = pd.read_parquet(RUN_DIR / "assignments.parquet")
    scored = pd.read_parquet(RUN_DIR / "scored.parquet")

    baseline = make_policy_family(assignments, n_slots=CONFIG.n_slots, seed=CONFIG.seed)
    model_policies = make_model_policy_family(scored, n_slots=CONFIG.n_slots)
    policies = baseline if model_policies.empty else pd.concat([baseline, model_policies], ignore_index=True)
    policies.to_parquet(RUN_DIR / "policy_assignments.parquet")


def step_evaluate() -> None:
    """Оффлайн-оценка стратегий по randomized logs на test split."""
    scored = pd.read_parquet(RUN_DIR / "scored.parquet")
    policies = pd.read_parquet(RUN_DIR / "policy_assignments.parquet")

    evaluation = evaluate_policies(
        scored,
        policies,
        n_slots=CONFIG.n_slots,
        split="test",
        n_bootstrap=80,
        seed=CONFIG.seed,
    )
    evaluation.to_csv(RUN_DIR / "evaluation_test.csv", index=False)


# --- Сборка DAG (импорт Airflow только здесь, чтобы шаги выше были автономны) ---

try:
    from datetime import datetime, timedelta

    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        "owner": "ml",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="crosssell_extra_npv_synthetic",
        description="Иллюстративная оркестрация синтетического CrossSell extra NPV пайплайна",
        schedule="@monthly",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        default_args=default_args,
        tags=["crosssell", "uplift", "synthetic", "demo"],
    ) as dag:
        generate = PythonOperator(task_id="generate_data", python_callable=step_generate_data)
        score = PythonOperator(task_id="score_pairs", python_callable=step_score_pairs)
        policies = PythonOperator(task_id="build_policies", python_callable=step_build_policies)
        evaluate = PythonOperator(task_id="evaluate_policies", python_callable=step_evaluate)

        generate >> score >> policies >> evaluate

except ImportError:  # Airflow не установлен — файл остается импортируемым как обычный модуль.
    dag = None
