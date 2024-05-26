from ultralytics import YOLO
import os
#Load model
model = YOLO("yolov8n.pt")

#Train: Hierbij traint je de algoritme van YOLO.v8 om de afbeeldingen van je dataset te herkennen
results = model.train(data='config.yaml', epochs=10)

#Validation: Hierbij gaat je controleren dat je YOLO.v8 model goed werd getraind.
results = model.val()


