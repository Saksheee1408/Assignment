import os
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="Image Classification API")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

MODEL_PATH = "models/best_model_A.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")
    model = None

CLASSES = sorted(['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot'])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=400, content={"error": "Invalid file format. Only JPG and PNG are allowed."})
    
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        return JSONResponse(status_code=400, content={"error": "File size exceeds 5MB limit."})
    
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded. Train the model first."})
        
    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid or empty image."})
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_resized = cv2.resize(img_rgb, (32, 32))
        img_normalized = img_resized.astype('float32') / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)
        
        predictions = model.predict(img_expanded)[0]
        
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        confidences = {CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "all_confidences": confidences
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
