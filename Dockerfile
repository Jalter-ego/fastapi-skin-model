# Usa una imagen base de Python ligera
FROM python:3.11-slim

ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema necesarias para TensorFlow y matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir matplotlib shap

COPY main.py .
COPY model/ ./model/

RUN mkdir -p /tmp/shap_images

# Exponer puerto
EXPOSE $PORT

# Comando para ejecutar la aplicación
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1"]
