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
model_mangga = tf.keras.models.load_model("model_mangga.h5")
model_mangga.trainable = False

# Label for mango classification
classes = ['Bukan Mangga','Terlalu Matang', 'Matang', 'Belum Matang']

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

def predict_mango_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((64, 64))  # Sesuai ekspektasi model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Jadi shape (1, 64, 64, 3)

    prediction = model_mangga.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_label = classes[pred_index]
    confidence = float(np.max(prediction))
    print(prediction)
    return pred_label, confidence

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

@app.post("/mango-prediction")
async def mango_prediction(file: UploadFile = File(...)):
    image_bytes = await file.read()

    label, confidence = predict_mango_from_bytes(image_bytes)
    return JSONResponse(content={
        "prediction": label,
        "confidence": f"{confidence:.2%}",
    })
