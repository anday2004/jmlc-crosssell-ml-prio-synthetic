# Synthetic CrossSell Extra NPV

Публичная синтетическая реализация проекта по персонализированному ранжированию CrossSell-предложений под ограничения NDA.

Репозиторий воспроизводит инженерную логику, а не рабочие данные или производственный код:

- генерация пользователей, историй событий, доступных категорий, рандомизированных показов и фактов утилизации;
- обучение табличной модели отклика, S-learner для uplift, обучаемого mini-transformer encoder по событиям и stacking-модели;
- ранжирование наборов категорий с учетом доступности и правила разнообразия по `group_id`;
- сравнение случайной стратегии, выбора по ценности, ранжирования по отклику, ранжирования по extra NPV и синтетического оракула;
- оценка стратегий через IPS/SNIPS, ESS и бутстреп-интервалы.

## Структура репозитория

```text
synthetic_crosssell_extra_npv_repo/
├── README.md
├── requirements.txt
├── demo.py
├── run_experiment.py
├── article/
│   ├── crosssell_extra_npv_public.tex
│   └── crosssell_extra_npv_public.pdf
├── docs/
│   ├── event_history_contract.md
│   ├── final_demo_results.md
│   ├── methodology.md
│   └── nda_boundary.md
├── notebooks/
│   ├── demo.ipynb
│   └── README.md
├── skills/
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
    └── test_pipeline.py
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
4. Сгенерировать потенциальные исходы `p0`, `p1`, наблюдаемую утилизацию и синтетический эффект для оракула.
5. Обучить табличную модель отклика и S-learner для uplift: одна outcome-модель получает признаки пары пользователь--категория и флаг показа `shown_feature`.
6. Обучить mini-transformer encoder истории клиента: события превращаются в токены, проходят через self-attention и attention pooling, затем объединяются с embedding категории.
7. Обучить stacking-модель, которая использует табличные признаки и трансформерные прогнозы.
8. Проскорить все пары пользователь--категория и жадно собрать топ-3 категории с разными `group_id`.
9. Оценить стратегии через IPS/SNIPS и сравнение с синтетическим оракулом.

## Стратегии

В `utils/policies.py` разделены скоринг и выбор:

- `response_model`: приоритизация по `p(utilization) * NPV`;
- `extra_npv_model`: приоритизация по `NPV * (p1 - p0)`;
- `response_transformer` и `extra_npv_transformer`: те же стратегии, но на прогнозах трансформерной ветки;
- `response_stacked` и `extra_npv_stacked`: стратегии на stacking-модели;
- `random`, `npv`, `response_true`, `oracle_extra_npv`: базовые и оракульные стратегии для проверки качества на синтетике.

Стратегия выбирает категории жадно: берет лучшую доступную категорию, удаляет из преселекта все категории с тем же `group_id`, затем выбирает следующую до топ-3.

## Как читать результаты

В синтетике есть несколько типов качества:

- `IPS/SNIPS`: category-level оценка новой стратегии по randomized logs. Она использует совпадения отдельных выбранных категорий с логом и обычно имеет лучший overlap;
- `set_IPS/set_SNIPS`: более строгая set-level оценка, где совпадение засчитывается только если вся top-3 выдача совпала со случайно показанным набором;
- `DR`: диагностическая оценка в стиле doubly robust на отложенном периоде; она комбинирует randomized logs и прогноз ценности;
- `true_extra_npv_value`: синтетический oracle-сигнал, доступный только потому, что генератор знает истинные `p0` и `p1`.

Основная таблица качества считается на `test`-периоде. Из-за конечного размера randomized logs oracle не обязан быть первым по IPS/SNIPS в каждом маленьком запуске. Зато он должен быть сильным по `true_extra_npv_value`, а `random` должен заметно проигрывать модельным стратегиям. В дефолтном `notebooks/demo.ipynb` обычно получается именно такая картина: random внизу, модельные стратегии выше, oracle задает верхнюю синтетическую границу.

Для дефолтного demo (`seed=42`) типичный sanity check:

- `random`: SNIPS около `61`, true extra NPV около `26`;
- `extra_npv_transformer`: SNIPS около `105`, true extra NPV около `41`;
- `extra_npv_stacked`: SNIPS около `104`, true extra NPV около `41`;
- `oracle_extra_npv`: максимальный true extra NPV около `45`.

Финальная таблица результатов сохранена в [`docs/final_demo_results.md`](docs/final_demo_results.md).

## Мой вклад / Contribution

В публичной версии я реализовал весь воспроизводимый контур проекта: генератор синтетических данных, randomized logs, модели отклика и uplift, mini-transformer encoder по событиям, stacking, слой выбора с ограничениями, IPS/SNIPS/DR-оценку, метрики, тесты, demo-запуск и NDA-safe документацию.

В реальной рабочей постановке мой вклад соответствует ML-части end-to-end: формализация таргета и единицы решения, подготовка исторических данных, обучение и сравнение моделей, скоринг всех пар клиент--категория, переход от response ranking к extra NPV логике, оффлайн-оценка стратегий и согласование ограничений раздачи с продуктовыми правилами.

## Навыки ИИ

В папке `skills/` лежат проектные навыки, которые формализуют помощь ИИ по этапам ML-работы:

- `ai-analyst-workflow`: анализ метрик, логов, дашбордов и подготовка задач/отчетов с подтверждением пользователя;
- `jmlc-adaption-council`: проверка конкурсной версии проекта с учетом ML, causal, product, MLOps и NDA;
- `ml-data-workflow`: данные, временные срезы, event history и утечки;
- `ml-code-workflow`: воспроизводимый код, demo, notebook и тесты;
- `ml-model-improvement`: улучшение бустинга, transformer, stacking и диагностик;
- `uplift-policy-workflow`: S-learner, extra NPV, randomized logs, category-level и set-level IPS.

## Что смотреть в коде

- `utils/data.py`: синтетика, randomized logs, потенциальные исходы и sequence table для трансформера;
- `utils/models.py`: табличный бустинг, обучаемый mini-transformer encoder, S-learner и stacking;
- `utils/policies.py`: greedy top-3 policy с ограничением на уникальный `group_id`;
- `utils/evaluation.py`: IPS/SNIPS, out-of-time DR-style diagnostic, ESS и bootstrap;
- `utils/metrics.py`: AUC/AP/Brier и uplift alignment;
- `notebooks/demo.ipynb`: основной интерактивный запуск для ревью;
- `demo.py`: резервный терминальный запуск того же пайплайна;
- `skills/`: навыки для системной работы с данными, кодом, моделями, uplift и NDA-safe адаптацией.

## Запуск

Основной способ ревью проекта — Jupyter notebook:

```bash
jupyter lab notebooks/demo.ipynb
```

В ноутбуке нужно выполнить `Run All`. Он сохраняет таблицы в
`artifacts/notebook_demo` и показывает test-split результаты:

- проверки качества синтетики;
- топ стратегий по SNIPS;
- AUC/AP/Brier для модельных голов;
- alignment predicted uplift с синтетическим истинным эффектом.

Если Jupyter не установлен:

```bash
pip install -r requirements.txt
```

Резервный терминальный запуск без Jupyter:

```bash
python demo.py
```

Он запускает тот же пайплайн и сохраняет артефакты в `artifacts/demo`.

Полный настраиваемый терминальный пайплайн:

```bash
python run_experiment.py --users 80 --categories 12 --periods 6 --seed 42 --artifacts-dir artifacts/run_01
```

Чтобы сохранить диагностические графики:

```bash
python run_experiment.py --users 80 --categories 12 --periods 6 --seed 42 --plots-dir artifacts/plots
```

Основные файлы в `artifacts/notebook_demo` или `artifacts/run_01`:

- `README.md`: короткая сводка запуска и топ стратегий;
- `evaluation.csv` и `evaluation_test.csv`: test-split category-level IPS/SNIPS, set-level IPS/SNIPS, DR, ESS, бутстреп-интервалы и синтетическое oracle-сравнение;
- `evaluation_all.csv`: такая же диагностика на всех split для sanity check;
- `model_quality.csv`: AUC, Average Precision и Brier для модельных голов;
- `uplift_alignment.csv`: корреляция predicted uplift с синтетическим истинным эффектом;
- `policy_assignments.csv`: какие категории выбрала каждая стратегия;
- `policy_slot_counts.csv`: проверка топ-3 и уникальности `group_id`;
- `model_score_summary.csv`: распределения прогнозов моделей;
- `transformer_sequence_sample.csv`: пример токенизированной истории без будущих событий.

Запуск тестов:

```bash
pytest
```
