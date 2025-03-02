import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from io import BytesIO
from pydantic import BaseModel
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Load your model (example with PyTorch)
# Change this part depending on your model type (YOLOv5, other PyTorch model, TensorFlow)
model = YOLO("./model/best.pt")

# Endpoint to handle image uploads and process the image
@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as a byte stream
        
        image_bytes = await file.read()
        print("image_bytes")
        image = Image.open(BytesIO(image_bytes))

        # Run inference with the loaded model (example with YOLOv5)
        results = model(image)  # YOLOv5 example

        # Convert results to JSON format (customize based on your model output)
        detections = results.pandas().xywh[0].to_dict(orient="records")

        # Save the output image with bounding boxes
        output_path = f"./output_images/{file.filename}"
        results.save(output_path)  # Saves the image with annotations

        return JSONResponse(content={
            "file_path": f"/output_images/{file.filename}",
            "detections": detections
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Error processing image", "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
