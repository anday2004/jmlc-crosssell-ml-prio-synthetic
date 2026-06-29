# Ноутбуки

Главный интерактивный запуск проекта:

```text
notebooks/demo.ipynb
```

Ноутбук является основным способом ревью проекта. Он вызывает тот же
код пайплайна, что и `demo.py`, но показывает результаты в удобном
пошаговом виде:

1. настройка окружения;
2. генерация синтетических данных;
3. обучение табличной модели, mini-transformer encoder и stacking;
4. оценка на test split стратегий через category-level IPS/SNIPS, set-level IPS/SNIPS и DR;
5. sanity checks;
6. model quality и uplift alignment;
7. сохранение CSV-артефактов и PNG-графиков.

Артефакты по умолчанию сохраняются в:

```text
artifacts/notebook_demo
```

Терминальный запуск `python demo.py` остается резервным способом воспроизвести
тот же пайплайн без Jupyter.
