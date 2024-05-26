from ultralytics import YOLO
import os

# Pad naar het getrainde YOLO-mode
model_path = os.path.join("runs/detect/train/weights/best.pt")

# Laad het model
model = YOLO(model_path) 

# Pad naar de map met alle test foto's
source = os.path.join('data/images/test')

# Maak een opslag directory voor resultaten
save_dir = os.path.join('results/')
os.makedirs(save_dir, exist_ok=True)

# Voer objectdetectie uit op elk bestand in de map
results = model(source, stream=True, save=True, project=save_dir)  # Opslag directory

# Resultaten laten zien
for result in results:
    print(result)

