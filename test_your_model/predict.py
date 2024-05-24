from ultralytics import YOLO

model_path = "C:\\Users\\jorge\\Documents\\Anaconda files\\Code\\runs\\detect\\train58\\weights\\best.pt"
#Load model
model = YOLO(model_path) 

# Perform tracking with the model

results = model.track(source="https://www.youtube.com/watch?v=hZYQYetndvs", conf=0.45, save=True, show=True)  