# Synthetic CrossSell ML Project

[![CI](https://github.com/anday2004/jmlc-crosssell-ml-prio-synthetic/actions/workflows/ci.yml/badge.svg)](https://github.com/anday2004/jmlc-crosssell-ml-prio-synthetic/actions/workflows/ci.yml)

Публичная синтетическая реализация проекта по персонализированному
ранжированию CrossSell-предложений под ограничения NDA для JMLC в ИТМО.

Репозиторий воспроизводит инженерную и ML-логику задачи, а не рабочие данные
или производственный код.

Презентация защиты (NDA-safe CrossSell-версия) — в [`presentation/`](presentation/crosssell_presentation.pdf).

## Результаты с первого взгляда (demo-конфигурация, `seed=42`)

Стратегия максимизирует не наблюдаемую утилизацию, а **инкремент от показа**
(extra NPV — добавленные деньги). Поэтому главная метрика — `true extra NPV`
(истинный эффект), а `SNIPS` (ценность отклика) — вторичная. `synthetic_ceiling`
задает потолок, `random` — нижнюю границу, а среди обучаемых стратегий лучшая —
`extra_npv_transformer` (она же лучшая по uplift alignment).

| Стратегия | true extra NPV (главная) | SNIPS (отклик) |
|---|---:|---:|
| `synthetic_ceiling` *(синтетический потолок)* | **44.96** | 78.37 |
| **`extra_npv_transformer` (лучшая обучаемая)** | **38.97** | 72.21 |
| `response_transformer` | 38.76 | 75.50 |
| `extra_npv_stacked` | 38.62 | 68.20 |
| `random` *(нижняя граница)* | 23.78 | 49.27 |

![Сравнение стратегий](docs/img/policy_comparison.png)

![Распределения скоров моделей](docs/img/score_distributions.png)

По `SNIPS` (ценность отклика) впереди response-стратегии — это ожидаемо: extra NPV
сознательно жертвует частью наблюдаемой утилизации (показывает не самым склонным,
а самым «сдвигаемым») ради добавленной ценности. Полная таблица всех десяти
стратегий, uplift alignment и overlap-проверка — в
[`docs/final_demo_results.md`](docs/final_demo_results.md). Графики детерминированы
(`seed=42`) и пересобираются командой из раздела [Запуск](#запуск).

## Почему модели написаны с нуля

Бустинг, mini-transformer encoder, S-learner и stacking реализованы вручную на
`numpy`/`pandas`, без `scikit-learn`, `xgboost` или `torch`. Это осознанный
выбор для публичной версии: репозиторий тривиально воспроизводится и ревьюится,
ничего не спрятано за черным ящиком библиотеки, а зависимости минимальны и
NDA-безопасны (не тянут внутренний стек). Корректность ключевых метрик
(`roc_auc`, `average_precision`, `brier`) сверена со scikit-learn в
[`tests/test_metrics_numeric.py`](tests/test_metrics_numeric.py).

## Основная идея

Проект моделирует задачу персонального выбора нескольких CrossSell-категорий
для клиента. Для каждого пользователя сначала скорируются все пары
пользователь--категория, после чего из доступного преселекта собирается top-3
набор с ограничением: в одной выдаче не должно быть двух категорий из одного
`group_id`.

В открытой версии реализованы три модельных слоя:

- табличный бустинг на агрегированных признаках пользователя и категории;
- mini-transformer encoder по историческим событиям клиента;
- stacking-модель, которая использует прогнозы трансформера как дополнительные признаки.

Трансформерная часть важна отдельно от бустинга. Табличная модель видит уже
собранные агрегаты, а трансформер получает историю событий клиента:
клики, транзакции, действия в приложении и прошлые CrossSell-взаимодействия.
Эти события превращаются в последовательность токенов, проходят через
self-attention, а затем объединяются с embedding категории. Такой подход
позволяет учитывать порядок, давность и контекст событий, которые сложно
полностью описать ручными табличными признаками.

В задаче сравниваются два подхода к ранжированию:

- `response ranking`: приоритизация по `p(utilization) * NPV`;
- `extra NPV ranking`: приоритизация по `NPV * (p1 - p0)`.

Для оценки используются randomized logs. Они нужны, чтобы сравнивать новую
политику ранжирования с базовыми стратегиями без прямого запуска на всей
аудитории.

В репозитории нет рабочих данных, внутреннего кода, настоящих категорий,
метрик или конфигураций. Вместо этого реализован воспроизводимый
синтетический контур:

- генерация пользователей, историй событий, доступных категорий, рандомизированных показов и фактов утилизации;
- обучение табличного бустинга, S-learner для uplift, mini-transformer encoder по событиям и stacking-модели;
- скоринг всех пар пользователь--категория;
- выбор top-3 категорий с учетом доступности и запрета на повтор `group_id`;
- сравнение random, NPV-only, response ranking, extra NPV и синтетического потолка;
- оффлайн-оценка стратегий по randomized logs.

## Структура репозитория

```text
synthetic_crosssell_extra_npv_repo/
├── README.md
├── requirements.txt
├── requirements-airflow.txt
├── Dockerfile
├── run_experiment.py
├── .github/workflows/
│   └── ci.yml
├── dags/
│   └── crosssell_dag.py
├── article/
│   ├── crosssell_extra_npv_public.tex
│   └── crosssell_extra_npv_public.pdf
├── presentation/
│   ├── crosssell_presentation.pptx
│   └── crosssell_presentation.pdf
├── docs/
│   ├── event_history_contract.md
│   ├── final_demo_results.md
│   ├── methodology.md
│   ├── nda_boundary.md
│   └── img/
│       ├── policy_comparison.png
│       └── score_distributions.png
├── notebooks/
│   ├── demo.ipynb
│   └── README.md
├── skills/
│   ├── example_council_review.md
│   ├── ai-analyst-workflow/
│   ├── jmlc-adaption-council/
│   ├── ml-code-workflow/
│   ├── ml-data-workflow/
│   ├── ml-model-improvement/
│   └── uplift-policy-workflow/
├── utils/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── metrics.py
│   ├── policies.py
│   ├── evaluation.py
│   └── plots.py
└── tests/
    ├── test_pipeline.py
    └── test_metrics_numeric.py
```

## Контракт данных

В синтетическом проекте используются четыре основные таблицы:

| Таблица | Гранулярность | Назначение |
|---|---|---|
| `users` | одна строка на пользователя | стабильные признаки пользователя и скрытые синтетические свойства |
| `categories` | одна строка на категорию | NPV, `group_id`, эмбеддинги, признаки доступности |
| `events` | одна строка на историческое событие | клики, транзакции, действия в приложении, прошлые CrossSell-события |
| `assignments` | одна строка на пользователя, период и категорию | рандомизированный показ, вероятность назначения, факт утилизации |

Самый важный проектный документ: [`docs/event_history_contract.md`](docs/event_history_contract.md). В нем описано, как должны генерироваться исторические события и как их преобразовывать в последовательности для трансформера без утечки таргета.

Дополнительно:

- [`docs/methodology.md`](docs/methodology.md): постановка, модели, стратегии и IPS/SNIPS-оценка;
- [`docs/nda_boundary.md`](docs/nda_boundary.md): что именно синтетическое и какие детали намеренно не раскрываются.

## Реализованный пайплайн

1. Сгенерировать синтетических пользователей, категории и исторические события.
2. Сформировать преселект доступных категорий для каждой пары пользователь--период.
3. Провести рандомизированный показ внутри тех же продуктовых ограничений, которые будут использоваться стратегиями.
4. Сгенерировать потенциальные исходы `p0`, `p1`, наблюдаемую утилизацию и синтетический эффект для потолка.
5. Обучить табличную модель отклика и S-learner для uplift: одна outcome-модель получает признаки пары пользователь--категория и флаг показа `shown_feature`.
6. Обучить mini-transformer encoder истории клиента: события превращаются в токены, проходят через self-attention и attention pooling, затем объединяются с embedding категории.
7. Обучить stacking-модель, которая использует табличные признаки и трансформерные прогнозы.
8. Проскорить все пары пользователь--категория и жадно собрать топ-3 категории с разными `group_id`.
9. Оценить стратегии через IPS/SNIPS и сравнение с синтетическим потолком.

## Стратегии

В `utils/policies.py` разделены скоринг и выбор:

- `response_model`: приоритизация по `p(utilization) * NPV`;
- `extra_npv_model`: приоритизация по `NPV * (p1 - p0)`;
- `response_transformer` и `extra_npv_transformer`: те же стратегии, но на прогнозах трансформерной ветки;
- `response_stacked` и `extra_npv_stacked`: стратегии на stacking-модели;
- `random`, `npv`, `response_true`, `synthetic_ceiling`: базовые и граничные стратегии для проверки качества на синтетике.

Стратегия выбирает категории жадно: берет лучшую доступную категорию, удаляет из преселекта все категории с тем же `group_id`, затем выбирает следующую до топ-3.

## Как читать результаты

В синтетике есть несколько типов качества:

- `true_extra_npv_value` (**главная метрика**): синтетический сигнал потолка истинного инкремента от показа — именно его максимизирует extra NPV-ранжирование;
- `SNIPS/IPS` (**вторичная**): оценка ценности *отклика* (наблюдаемой утилизации) по randomized logs; по ней естественно лидируют response-стратегии;
- `DR`: диагностическая оценка в стиле doubly robust на отложенном периоде;
- `roc_auc_shown`: causal overlap-проверка — насколько факт показа предсказуем по предраздаточным признакам (должна быть ≈0.5).

Главная метрика — добавленная ценность (`true_extra_npv_value`), потому что цель раздачи — инкремент, а не наблюдаемая утилизация. По ней ожидаемая картина: `synthetic_ceiling` задает потолок, `random` — низ, а среди обучаемых стратегий лучшая — `extra_npv_transformer`. По `SNIPS` (ценность отклика) впереди response-стратегии — это не недостаток, а суть extra NPV: жертвовать частью наблюдаемой утилизации ради добавленной ценности.

Для demo (`seed=42`) типичный sanity check (по `true_extra_npv_value`):

- `synthetic_ceiling`: максимум, около `45` (синтетический потолок);
- `extra_npv_transformer`: лучшая обучаемая, около `39`, и лучший uplift alignment (corr ≈ `0.58`);
- `random`: внизу, около `24`;
- overlap `roc_auc_shown` ≈ `0.49–0.50` — рандомизация чистая.

> **Упрощенность demo.** Бустинг и трансформер — намеренно облегченные mini-реализации на numpy: они показывают принцип и инженерный контур, а не рабочие метрики. Абсолютные значения синтетические; в рабочем контуре модели полноразмерные, и относительная сила моделей может отличаться.

Финальная таблица результатов сохранена в [`docs/final_demo_results.md`](docs/final_demo_results.md).

## Мой вклад / Contribution

В публичной версии я реализовал весь воспроизводимый контур проекта: генератор синтетических данных, randomized logs, модели отклика и uplift, mini-transformer encoder по событиям, stacking, слой выбора с ограничениями, IPS/SNIPS/DR-оценку, метрики, тесты, воспроизводимую инженерную обвязку (CLI, Dockerfile, GitHub Actions, иллюстративный Airflow DAG) и NDA-safe документацию.

В реальной рабочей постановке мой вклад соответствует ML-части end-to-end: формализация таргета и единицы решения, подготовка исторических данных, обучение и сравнение моделей, скоринг всех пар клиент--категория, переход от response ranking к extra NPV логике, оффлайн-оценка стратегий и согласование ограничений раздачи с продуктовыми правилами.

## Навыки ИИ

В папке `skills/` лежат проектные навыки, которые формализуют помощь ИИ по этапам ML-работы:

- `ai-analyst-workflow`: анализ метрик, логов, дашбордов и подготовка задач/отчетов с подтверждением пользователя;
- `jmlc-adaption-council`: проверка конкурсной версии проекта с учетом ML, causal, product, MLOps и NDA;
- `ml-data-workflow`: данные, временные срезы, event history и утечки;
- `ml-code-workflow`: воспроизводимый код, demo, notebook и тесты;
- `ml-model-improvement`: улучшение бустинга, transformer, stacking и диагностик;
- `uplift-policy-workflow`: S-learner, extra NPV, randomized logs и проверка policy evaluation.

Два разобранных прогона навыка `jmlc-adaption-council` по этому репозиторию (вход → находки → внесенное исправление) лежат в [`skills/example_council_review.md`](skills/example_council_review.md).

## Что смотреть в коде

- `utils/data.py`: синтетика, randomized logs, потенциальные исходы и sequence table для трансформера;
- `utils/models.py`: табличный бустинг, обучаемый mini-transformer encoder, S-learner и stacking;
- `utils/policies.py`: greedy top-3 policy с ограничением на уникальный `group_id`;
- `utils/evaluation.py`: IPS/SNIPS, out-of-time DR-style diagnostic, ESS и bootstrap;
- `utils/metrics.py`: AUC/AP/Brier и uplift alignment;
- `notebooks/demo.ipynb`: основной интерактивный запуск для ревью;
- `run_experiment.py`: параметризуемый терминальный запуск того же пайплайна (`run_pipeline`);
- `dags/crosssell_dag.py`: иллюстративная Airflow-оркестрация поверх тех же step-функций;
- `Dockerfile`, `.github/workflows/ci.yml`: воспроизводимая сборка и CI (тесты на каждый push);
- `skills/`: навыки для системной работы с данными, кодом, моделями, uplift и NDA-safe адаптацией.

## Запуск

У проекта три входа поверх одного движка (`run_pipeline` в `run_experiment.py`):
интерактивный notebook, CLI и Airflow DAG. Логика нигде не дублируется.

```bash
pip install -r requirements.txt
```

**1. Notebook (основной способ ревью).**

```bash
jupyter lab notebooks/demo.ipynb
```

Выполнить `Run All`. Сохраняет таблицы в `artifacts/notebook_demo` и показывает
test-split результаты: проверки синтетики, топ стратегий по SNIPS, AUC/AP/Brier
для модельных голов и alignment predicted uplift с истинным эффектом.

**2. CLI (параметризуемый, для воспроизводимости).**

```bash
python run_experiment.py --users 80 --categories 12 --periods 6 --seed 42 --artifacts-dir artifacts/run_01
```

Чтобы пересобрать диагностические графики (в т.ч. картинки для README):

```bash
python run_experiment.py --users 80 --categories 12 --periods 6 --seed 42 --plots-dir docs/img
```

**3. Airflow DAG (иллюстрация оркестрации).** Шаги `generate → score → policies →
evaluate` вызывают те же функции из `utils/`. Это публичная параллель к
продакшн-оркестрации, которая под NDA живет в закрытом контуре (GitLab + Airflow).

```bash
pip install -r requirements-airflow.txt   # только для DAG
# затем зарегистрировать dags/crosssell_dag.py в своем Airflow
```

## Docker и CI

Образ собирается и реально запускает пайплайн с тестами:

```bash
docker build -t crosssell-extra-npv .
docker run --rm crosssell-extra-npv
```

CI на GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
на каждый push прогоняет `pytest`, smoke-запуск пайплайна и сборку Docker-образа;
статус виден по бейджу в шапке README.

Основные файлы в `artifacts/notebook_demo` или `artifacts/run_01`:

- `README.md`: короткая сводка запуска и топ стратегий;
- `evaluation.csv` и `evaluation_test.csv`: IPS/SNIPS, DR, ESS, бутстреп-интервалы и синтетическое сравнение с потолком;
- `evaluation_all.csv`: такая же диагностика на всех split для sanity check;
- `model_quality.csv`: AUC, Average Precision и Brier для модельных голов;
- `uplift_alignment.csv`: корреляция predicted uplift с синтетическим истинным эффектом;
- `propensity_overlap.csv`: causal overlap-проверка — ROC AUC предсказания факта показа по предраздаточным признакам (≈0.5 = рандомизация чистая);
- `policy_assignments.csv`: какие категории выбрала каждая стратегия;
- `policy_slot_counts.csv`: проверка топ-3 и уникальности `group_id`;
- `model_score_summary.csv`: распределения прогнозов моделей;
- `transformer_sequence_sample.csv`: пример токенизированной истории без будущих событий.

Запуск тестов:

```bash
pytest
```
