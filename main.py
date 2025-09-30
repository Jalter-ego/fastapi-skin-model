from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io 
import shap
import uuid
import os

# --- 1. Inicializar la aplicación FastAPI ---
app = FastAPI(
    title="API de Clasificación de Piel",
    description="Despliega un modelo de TensorFlow para clasificar condiciones de la piel a partir de imágenes.",
    version="1.0.0"
)

# --- 2. Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://happy-dune-00983ce1e.1.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/modelo_skin.h5"
model = None 
CLASSES = ["melanoma", "normal_skin", "psoriasis"] 

MODEL_PATH2 = "model/modelo_melanoma_benign-malignant.h5"
model2 = None
CLASSES2 = ["benigno", "maligno"]

explainer = None


@app.on_event("startup")
async def load_models_on_startup():
    global model, model2
    try:
        print(f"Cargando modelo desde: {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modelo 1 cargado exitosamente.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo 1. Detalles: {e}")
        raise RuntimeError(f"Error al cargar el modelo 1: {e}")
    try:
        print(f"Cargando modelo desde: {MODEL_PATH2}...")
        model2 = tf.keras.models.load_model(MODEL_PATH2)
        print("Modelo 2 cargado exitosamente.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo 2. Detalles: {e}")
        raise RuntimeError(f"Error al cargar el modelo 2: {e}")

@app.get("/")
async def read_root():
    return {"message": "API de Clasificación de Piel está corriendo!"}

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    """
    Recibe una imagen (formato de archivo) y devuelve la clase predicha
    y la confianza del modelo de clasificación de piel.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo proporcionado no es una imagen.")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        pil_image = pil_image.resize((224, 224)) # Redimensionar
        img_array = img_to_array(pil_image)      # Convertir a array de NumPy
        img_array = img_array / 255.0            # Normalizar
        img_array = np.expand_dims(img_array, axis=0) # Añadir dimensión de batch

        predictions = model.predict(img_array)[0] # [0] para obtener las probabilidades del primer (y único) elemento del batch
        class_index = np.argmax(predictions)
        class_label = CLASSES[class_index]
        confidence = float(predictions[class_index])

        return JSONResponse(content={
            "class": class_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")


@app.post("/predict_benign_malignant/")
async def predict_benign_malignant(image: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve la clase predicha (benigno/maligno)
    y la confianza del modelo de melanoma.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo proporcionado no es una imagen.")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        pil_image = pil_image.resize((224, 224))
        img_array = img_to_array(pil_image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model2.predict(img_array)[0]
        class_index = np.argmax(predictions)
        class_label = CLASSES2[class_index]
        confidence = float(predictions[class_index])

        return JSONResponse(content={
            "class": class_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

# Para ejecutar localmente, usa: uvicorn main:app --reload --host 0.0.0.0 --port 5000
# Puedes acceder a la documentación interactiva en: http://127.0.0.1:8000/docs