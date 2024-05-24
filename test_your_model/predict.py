from ultralytics import YOLO
import os

model_path = os.path.join("runs/detect/train/weights/best.pt")

#Load model
model = YOLO(model_path) 

# Define the save directory for results
save_dir = os.path.join('results/')
os.makedirs(save_dir, exist_ok=True)

# Perform tracking with the model

results = model.track(source="https://www.youtube.com/watch?v=jIGKgZPMYxI", conf=0.45, save=True, show=True, project=save_dir)  