from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
from PIL import Image
import io
import os
from ultralytics import YOLO
from typing import List, Dict
import json

app = FastAPI(title="BlurAI API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save original image
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        cv2.imwrite(file_path, img)
        
        # Run object detection
        results = model(img)
        
        # Process detection results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "size": {
                        "width": float(x2 - x1),
                        "height": float(y2 - y1)
                    }
                })
        
        return {
            "filename": file.filename,
            "detections": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/apply-effect")
async def apply_effect(
    filename: str,
    effect_type: str,
    target_objects: List[str] = None,
    global_effect: bool = False
):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Read image
        img = cv2.imread(file_path)
        
        if global_effect:
            if effect_type == "blur_faces":
                # Run face detection and blur
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    face_roi = img[y:y+h, x:x+w]
                    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    img[y:y+h, x:x+w] = blurred_face
                    
        else:
            # Apply effect to specific objects
            results = model(img)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]
                    
                    if class_name in target_objects:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        if effect_type == "blur":
                            roi = img[y1:y2, x1:x2]
                            blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                            img[y1:y2, x1:x2] = blurred
                        elif effect_type == "delete":
                            img[y1:y2, x1:x2] = 255  # White out
        
        # Save processed image
        output_path = os.path.join(UPLOAD_DIR, f"processed_{filename}")
        cv2.imwrite(output_path, img)
        
        return {"processed_filename": f"processed_{filename}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_image(filename: str, high_res: bool = False):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    if high_res:
        # Implement high-resolution processing if needed
        pass
    
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 