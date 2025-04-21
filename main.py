from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI()
# Bolehkan semua origin (untuk testing / dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti ini ke domain frontend kamu kalau sudah production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model sekali saja (lebih efisien)
model = tf.keras.models.load_model("truck_classifier_best_model.h5")

def predict_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)[0][0]
    label = "Truck" if pred >= 0.5 else "Not Truck"
    confidence = float(pred if pred >= 0.5 else 1 - pred)
    return label, confidence

@app.post("/truck-recognition")
async def truck_recognition(file: UploadFile = File(...)):
    image_bytes = await file.read()
    label, confidence = predict_image_from_bytes(image_bytes)
    return JSONResponse(content={
        "prediction": label,
        "confidence": f"{confidence:.2%}"
    })

@app.get("/")
async def root():
    return {"message": "Welcome to the Truck Recognition API!"}
