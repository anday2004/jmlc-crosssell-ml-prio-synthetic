# Образ для воспроизводимого запуска синтетического CrossSell extra NPV пайплайна.
# В закрытом рабочем контуре расчет упакован в свой Docker-образ и оркестрируется
# через CI/Airflow; здесь публикуется облегченная, но реально собираемая версия
# того же инженерного контура поверх синтетических данных.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Сначала зависимости — кешируется отдельным слоем.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Затем код.
COPY . .

# По умолчанию контейнер прогоняет тесты и запускает demo-конфиг пайплайна,
# сохраняя CSV-артефакты в /app/artifacts/docker_run.
CMD ["sh", "-c", "pytest -q && python run_experiment.py --users 80 --categories 12 --periods 6 --seed 42 --artifacts-dir artifacts/docker_run"]
