# Навыки Codex

В этой папке лежат проектные навыки Codex. Они нужны не для запуска модели, а
для воспроизводимого ревью: чтобы одинаково проверять данные, код, uplift,
модели, NDA-границу и аналитические выводы.

У каждого навыка одинаковая структура:

```text
skill-name/
├── SKILL.md
└── agents/openai.yaml
```

`SKILL.md` содержит рабочие инструкции для агента. `agents/openai.yaml` хранит
метаданные для интерфейса: название, короткое описание и стартовый prompt.

## Список навыков

| Навык | Зачем нужен |
| --- | --- |
| `jmlc-adaption-council` | Строгое ревью конкурсных артефактов с разных сторон: ML, causal inference, продукт, инженерия, личный вклад и NDA. |
| `ml-data-workflow` | Проверка данных: табличная витрина, event history, временные срезы, randomized logs, propensities, доступность и преселекты. |
| `ml-code-workflow` | Проверка воспроизводимости кода: общий пайплайн, notebook, demo, тесты, артефакты и чистота репозитория. |
| `ml-model-improvement` | Улучшение моделей: бустинг, трансформер, stacking, калибровка, uplift alignment и policy-level диагностики. |
| `uplift-policy-workflow` | Проверка uplift-части: S-learner, extra NPV, randomized logs, overlap, IPS/SNIPS и top-N ограничения. |
| `ai-analyst-workflow` | Сценарий ИИ-аналитика: анализ метрик, логов и дашбордов, поиск аномалий, подготовка задач и отчетов с подтверждением пользователя. |

## Примеры вызова

```text
Use $jmlc-adaption-council to review the article and README for NDA safety.
Use $ml-data-workflow to check whether the synthetic event history contract is leakage-safe.
Use $uplift-policy-workflow to review the IPS/SNIPS evaluation logic.
Use $ml-code-workflow to verify that the demo, notebook, and tests are reproducible.
```

## Примеры реальной работы навыка

Два разобранных прогона `jmlc-adaption-council` по этому репозиторию (вход →
находки по линзам → внесенное исправление) лежат в
[`example_council_review.md`](example_council_review.md):

- engineering-линза нашла, что заявка на CI/Docker/Airflow не подтверждена кодом → добавлены Dockerfile, GitHub Actions и DAG;
- NDA-линза нашла имя работодателя в сдаваемой версии статьи → публичная версия обобщена до «крупной финтех-экосистемы».

Навыки не содержат рабочие данные, внутренние схемы таблиц, токены, адреса
сервисов или конфигурацию работодателя.
