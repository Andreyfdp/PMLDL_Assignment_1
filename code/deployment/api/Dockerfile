FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    scikit-learn \
    joblib

COPY code/deployment/api/main.py /app/code/deployment/api/main.py
COPY models/penguins_model.joblib /app/models/penguins_model.joblib

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "/app/code/deployment/api", "--host", "0.0.0.0", "--port", "8000"]