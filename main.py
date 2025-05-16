from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io # Para manejar los bytes de la imagen

# --- 1. Inicializar la aplicación FastAPI ---
app = FastAPI(
    title="API de Clasificación de Piel",
    description="Despliega un modelo de TensorFlow para clasificar condiciones de la piel a partir de imágenes.",
    version="1.0.0"
)

# --- 2. Configurar CORS (Crucial para que tu frontend se pueda comunicar) ---
# En desarrollo, permitimos todos los orígenes.
# En producción, deberías cambiar '*' por el dominio o dominios específicos de tu frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"], # Permite todos los métodos (POST en este caso)
    allow_headers=["*"], # Permite todos los encabezados
)

# --- 3. Cargar el Modelo de TensorFlow ---
# Define la ruta a tu modelo guardado.
MODEL_PATH = "model/modelo_skin.h5" # Asumiendo que está en una subcarpeta 'model'

model = None # Inicializamos model como None
CLASSES = ["melanoma", "normal_skin", "psoriasis"] # Clases de tu modelo

@app.on_event("startup")
async def load_model_on_startup():
    """
    Carga el modelo de TensorFlow cuando la aplicación se inicia.
    Esto asegura que el modelo solo se cargue una vez y esté disponible para todas las peticiones.
    """
    global model
    try:
        print(f"Cargando modelo desde: {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        # Opcional: imprimir el resumen del modelo para verificar que cargó correctamente
        # model.summary()
        print("Modelo cargado exitosamente.")
    except Exception as e:
        # Si el modelo no puede cargarse, levanta una excepción para que la aplicación no inicie
        print(f"ERROR: No se pudo cargar el modelo. Asegúrate de que la ruta sea correcta y el archivo exista. Detalles: {e}")
        raise RuntimeError(f"Error al cargar el modelo: {e}")

# --- 4. Endpoint Raíz (Opcional, para verificar que la API está viva) ---
@app.get("/")
async def read_root():
    return {"message": "API de Clasificación de Piel está corriendo!"}

# --- 5. Endpoint de Predicción ---
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    """
    Recibe una imagen (formato de archivo) y devuelve la clase predicha
    y la confianza del modelo de clasificación de piel.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo proporcionado no es una imagen.")

    try:
        # Leer el contenido del archivo de imagen
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocesamiento de la imagen, igual que en tu código Flask
        pil_image = pil_image.resize((224, 224)) # Redimensionar
        img_array = img_to_array(pil_image)      # Convertir a array de NumPy
        img_array = img_array / 255.0            # Normalizar
        img_array = np.expand_dims(img_array, axis=0) # Añadir dimensión de batch

        # Realizar la predicción
        predictions = model.predict(img_array)[0] # [0] para obtener las probabilidades del primer (y único) elemento del batch
        class_index = np.argmax(predictions)
        class_label = CLASSES[class_index]
        confidence = float(predictions[class_index])

        # Devolver la respuesta JSON
        return JSONResponse(content={
            "class": class_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        # Manejo de errores general para cualquier problema durante el procesamiento o predicción
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

# Para ejecutar localmente, usa: uvicorn main:app --reload --host 0.0.0.0 --port 5000
# Puedes acceder a la documentación interactiva en: http://127.0.0.1:5000/docs