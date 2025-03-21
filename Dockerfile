FROM python:3.12

WORKDIR /IIAgents

# Установка Poetry
RUN pip install poetry

# Копируем зависимости
COPY pyproject.toml poetry.lock ./

# Устанавливаем зависимости
RUN poetry install --no-root

# Копируем исходный код
COPY . .

# Открываем порт
EXPOSE 8000

# Запуск приложения через Poetry
CMD ["poetry", "run", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "src.api.service:app"]