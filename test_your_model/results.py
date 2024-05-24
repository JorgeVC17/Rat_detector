from ultralytics import YOLO
import os

model_path = os.path.join("runs/detect/train/weights/best.pt")
#Load model
model = YOLO(model_path) 

# Define path to directory containing images and videos for inference
source = os.path.join('data/images/test')

# Define the save directory for results
save_dir = os.path.join('results/')
os.makedirs(save_dir, exist_ok=True)

# Run inference on the source directory
results = model(source, stream=True, save=True, project=save_dir)  # generator of Results objects

# Process and display results
for result in results:
    # Print or log the result
    print(result)

