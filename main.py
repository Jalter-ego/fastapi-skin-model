from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io 
import os
from dotenv import load_dotenv

load_dotenv() 
from azure.storage.blob import BlobServiceClient 
import uuid 

app = FastAPI(
    title="API de Clasificación de Piel",
    description="Despliega un modelo de TensorFlow para clasificar condiciones de la piel a partir de imágenes.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://happy-dune-00983ce1e.1.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "ml-models" # El contenedor que creaste

# Nombres de los modelos en el Blob Storage
MODEL_BLOB_NAME_1 = "modelo_skin.h5"
MODEL_BLOB_NAME_2 = "modelo_melanoma_benign-malignant.h5"

# Rutas locales temporales donde se guardarán los modelos descargados
MODEL_PATH = f"./temp/{MODEL_BLOB_NAME_1}"
MODEL_PATH2 = f"./temp/{MODEL_BLOB_NAME_2}"

model = None 
CLASSES = ["melanoma", "normal_skin", "psoriasis"] 
model2 = None
CLASSES2 = ["benigno", "maligno"]

explainer = None

def download_blob_to_file(blob_service_client: BlobServiceClient, container_name, blob_name, local_file_path):
    """Descarga un blob y lo guarda en una ruta de archivo local."""
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        print(f"Descargando {blob_name} a {local_file_path}...")
        with open(file=local_file_path, mode="wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        print(f"Descarga de {blob_name} completada.")
        return True
    except Exception as e:
        print(f"ERROR al descargar el blob {blob_name}. Detalles: {e}")
        return False


@app.on_event("startup")
async def load_models_on_startup():
    global model, model2
    
    if not AZURE_STORAGE_CONNECTION_STRING:
        print("ERROR: La variable AZURE_STORAGE_CONNECTION_STRING no está configurada.")
        raise RuntimeError("Fallo al obtener la cadena de conexión de Azure Storage.")

    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        
        if download_blob_to_file(blob_service_client, CONTAINER_NAME, MODEL_BLOB_NAME_1, MODEL_PATH):
            print(f"Cargando modelo 1 desde: {MODEL_PATH}...")
            model = tf.keras.models.load_model(MODEL_PATH) 
            print("Modelo 1 cargado exitosamente.")
        else:
            raise RuntimeError("Fallo en la descarga y carga del Modelo 1.")
            
        if download_blob_to_file(blob_service_client, CONTAINER_NAME, MODEL_BLOB_NAME_2, MODEL_PATH2):
            print(f"Cargando modelo 2 desde: {MODEL_PATH2}...")
            model2 = tf.keras.models.load_model(MODEL_PATH2)
            print("Modelo 2 cargado exitosamente.")
        else:
            raise RuntimeError("Fallo en la descarga y carga del Modelo 2.")
            
    except Exception as e:
        print(f"ERROR crítico en la carga de modelos: {e}")
        raise RuntimeError(f"Error al inicializar la aplicación: {e}")



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