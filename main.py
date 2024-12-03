from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.image import img_to_array
from schemas.Predictor_Schema import PredictionResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import io

app = FastAPI(
    title="API - SISTEMAS INTELIGENTES",
    description="API para detectar y clasificar tumores por medio de imágenes de resonancia magnética."
)

# Cargar el modelo entrenado al iniciar la API
MODEL_PATH = "models_trained/trained_model_tumor.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
model = load_model(MODEL_PATH)


# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # de momento permitimos todos los origenes
    allow_credentials=True,
    allow_methods=["*"],  # métodos permitidos
    allow_headers=["*"],  # cabeceras permitidas
)

# redirecciona a la documentacion
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


# Clases del modelo (dataset)
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Endpoint para realizar predicciones
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Validar tipo de archivo
        if file.content_type not in ["image/jpeg", "image/png", "image/tiff"]:
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes JPEG, PNG o TIFF")
        
        # Leer la imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")  # Escala de grises
        image = image.resize((28, 28))  # Ajustar tamaño
        image_array = img_to_array(image).astype("float32") / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)  # Redimensionar para el modelo

        # Realizar predicción
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        # Responder con la clase predicha y su probabilidad
        result = PredictionResponse(
            tumor_class=CLASSES[predicted_class],
            confidence=float(confidence),
            # redondear a 2 decimales la probabilidad
            probabilidad=float(round(confidence * 100, 2))
        ).dict()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {e}")