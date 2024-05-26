import cv2
from ultralytics import YOLO
import os

# Pad naar het getrainde YOLO-model
model_path = os.path.join("runs/detect/train/weights/best.pt")
model = YOLO(model_path)

# Open de video bestand
video_path = os.path.join("test_videos/test_video1.mp4")
cap = cv2.VideoCapture(video_path)

# Oneindige loop om continu beelden/frames van de videos te verwerken
while cap.isOpened():
    # Lees een frame van de video
    success, frame = cap.read()

    if success:
        #  Voer objectdetectie uit op elk frame
        results = model(frame)

        # Visualiseert de resultaten op het frame
        annotated_frame = results[0].plot()

        # Genoteerd frame laten zien
        cv2.imshow("YOLOv8 Inference", annotated_frame)

         # Stop de loop af als de 'q'-toets wordt ingedrukt
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Stop de loop af aan het eind van de video
        break

# Display venster afsluiten
cap.release()
cv2.destroyAllWindows()