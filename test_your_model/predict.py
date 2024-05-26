from ultralytics import YOLO
import os

# Pad naar het getrainde YOLO-mode
model_path = os.path.join("runs/detect/train/weights/best.pt")

# Laad het model
model = YOLO(model_path) 

# Maak een opslag directory voor resultaten
save_dir = os.path.join('results/')
os.makedirs(save_dir, exist_ok=True)

# Uitvoer tracking met het model

results = model.track(source="https://www.youtube.com/watch?v=jIGKgZPMYxI", # Youtube video
                      conf=0.45,  # Confidence lager dan 45 wordt niet weergegeven
                      save=True, # Prediction wordt opgeslag
                      show=True, # Video in een venster laten zien
                      project=save_dir)  # Opslag directory