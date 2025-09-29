# Usa una imagen base de Python ligera
FROM python:3.11-slim

# Establecer variables de entorno
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

# Copiar requirements primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir matplotlib shap

# Copiar código fuente y modelos
COPY main.py .
COPY model/ ./model/

# Crear directorio para archivos temporales de SHAP
RUN mkdir -p /tmp/shap_images

# Exponer puerto
EXPOSE $PORT

# Comando para ejecutar la aplicación
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1"]
