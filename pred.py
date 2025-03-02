from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
import os
import cv2

print(os.path.exists("./yolo11_tf/saved_model.pb"))
model_path = os.path.abspath("./yolo11_tf")

app = FastAPI()

# Load the TensorFlow SavedModel (.pb format)
model = tf.saved_model.load(model_path)  # Update with your model path
print(model.signatures)
infer = model.signatures["serving_default"]  # Get inference function

# Load the image
image_path = "./img/DSC03518.jpg"  # Change to your image path
image = cv2.imread(image_path)

# Convert BGR (OpenCV default) to RGB (TensorFlow default)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to match model input (640x640)
image = cv2.resize(image, (640, 640))

# Normalize (if required by model)
image = image / 255.0  # If model expects values between 0 and 1

# Convert to tensor
input_tensor = np.expand_dims(image, axis=0).astype(np.float32)  # Shape: (1, 640, 640, 3)

# Run inference
output = infer(images=tf.convert_to_tensor(input_tensor))

# Assuming 'output' is the model output
detections = output['output_0'].numpy()[0]  # Remove batch dimension

# Extract bounding boxes
boxes = detections[:4, :]  # First 4 values are bbox info
scores = detections[4:, :]  # Remaining are class scores

# Convert box format (YOLO outputs center_x, center_y, width, height)
x_center, y_center, width, height = boxes[0], boxes[1], boxes[2], boxes[3]

# Convert to (x_min, y_min, x_max, y_max)
x_min = x_center - (width / 2)
y_min = y_center - (height / 2)
x_max = x_center + (width / 2)
y_max = y_center + (height / 2)

# Find the best class prediction
class_ids = np.argmax(scores, axis=0)  # Get class index with highest probability
confidences = np.max(scores, axis=0)  # Get confidence scores

# Filter predictions by confidence threshold
threshold = 0.5
valid_detections = confidences > threshold

# Final results
final_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)[valid_detections]
final_classes = class_ids[valid_detections]
final_confidences = confidences[valid_detections]

print("Final Bounding Boxes:", final_boxes)
print("Final Classes:", final_classes)
print("Final Confidences:", final_confidences)

class InputData(BaseModel):
    data: list  # Expecting an array of input features

@app.post("/predict")
async def predict(input_data: InputData):
    input_tensor = tf.convert_to_tensor(["lay"], dtype=tf.float32)  # Adjust dtype if needed

    # Run inference
    output = infer(input_tensor)
    
    # Extract prediction
    prediction = list(output.values())[0].numpy().tolist()

    return {"prediction": prediction}

# Start server: uvicorn predict_server:app --host 0.0.0.0 --port 8000
