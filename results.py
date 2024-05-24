from ultralytics import YOLO
import os

model_path = "C:\\Users\\jorge\\Documents\\Python\\Rat_detector\\runs\\detect\\train2\\weights\\best.pt"
#Load model
model = YOLO(model_path) 

# Define path to directory containing images and videos for inference
source = 'C:\\Users\\jorge\\Documents\\Python\\Rat_detector\\data\\images\\test'

# Define the save directory for results
save_dir = 'C:\\Users\\jorge\\Documents\\Python\\Rat_detector\\results\\'
os.makedirs(save_dir, exist_ok=True)

# Run inference on the source directory
results = model(source, stream=True, save=True, project=save_dir)  # generator of Results objects

# Process and display results
for result in results:
    # Print or log the result
    print(result)

