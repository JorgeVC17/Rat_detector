from ultralytics import YOLO
import os
#Load model
model = YOLO("yolov8n.pt")

#Use model
results = model.train(data='config.yaml', epochs=10)

#Validation
results = model.val()


