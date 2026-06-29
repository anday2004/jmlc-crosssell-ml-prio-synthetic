# Финальные результаты demo

Финальный запуск:

```bash
python demo.py --bootstrap 80 --artifacts-dir artifacts/final_submission
```

Конфигурация по умолчанию: `seed=42`, `users=80`, `categories=12`,
`periods=6`, `slots=3`.

## Policy evaluation на test split

| Стратегия | SNIPS | IPS | DR-style | True extra NPV |
|---|---:|---:|---:|---:|
| `extra_npv_transformer` | 105.38 | 91.43 | 102.80 | 40.74 |
| `response_transformer` | 104.67 | 97.01 | 105.02 | 40.64 |
| `extra_npv_stacked` | 103.63 | 91.43 | 101.83 | 40.60 |
| `response_true` | 102.84 | 98.53 | 104.79 | 42.35 |
| `oracle_extra_npv` | 102.44 | 81.22 | 99.31 | 45.24 |
| `npv` | 100.30 | 87.40 | 99.72 | 41.21 |
| `response_stacked` | 100.24 | 87.14 | 99.20 | 40.84 |
| `response_model` | 97.59 | 86.00 | 97.94 | 40.71 |
| `extra_npv_model` | 94.53 | 80.27 | 93.94 | 40.63 |
| `random` | 60.80 | 71.05 | 63.60 | 25.77 |

## Uplift alignment

| Модель | Корреляция с true extra NPV | Top-decile lift |
|---|---:|---:|
| `extra_npv_stacked` | 0.6757 | 1.9627 |
| `extra_npv_transformer` | 0.6752 | 1.9378 |
| `extra_npv_model` | 0.5711 | 1.8330 |

## Как читать

Главная sanity-проверка выполняется: `random` заметно хуже модельных стратегий,
а `oracle_extra_npv` задает верхнюю границу по синтетическому истинному эффекту.
Последовательностная ветка сильнее табличной в uplift alignment и дает лучший
model-based policy score по category-level SNIPS. Это отражает синтетический
мир, где добавочный эффект показа зависит от недавних event-паттернов клиента,
которые трансформер видит подробнее, чем табличные агрегаты.

`oracle_extra_npv` не является рабочей моделью: он использует скрытые истинные
`p0_true` и `p1_true`, доступные только в синтетике.
