FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app
RUN pip install --upgrade pip && pip install -e ".[assistant,api]"

EXPOSE 8000

CMD ["uvicorn", "chronos_sales.assistant_api:app", "--host", "0.0.0.0", "--port", "8000"]
