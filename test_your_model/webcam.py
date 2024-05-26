from ultralytics import YOLO
import cv2
import math 
import os

# Start de webcam en stel de resolutie in
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Stel de breedte van het beeld in op 640 pixels
cap.set(4, 480)  # Stel de hoogte van het beeld in op 480 pixels

# Pad naar het getrainde YOLO-model
model_path = os.path.join("runs/detect/train/weights/best.pt")
model = YOLO(model_path)  # Laad het model

# Objectklassen
classNames = ["Rat"]

# Oneindige loop om continu beelden/frames van de webcam te verwerken
while True:
    success, img = cap.read()  # Lees een frame van de webcam
    results = model(img, stream=True)  # Voer objectdetectie uit op elk frame

    # Coördinaten
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Haal de coördinaten van het bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # omzet naar gehele getallen

            # Teken het box op het frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Vertrouwbaarheid score 
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # Haal de klasse van het gedetecteerde object op
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]  # Positie voor de tekst
            font = cv2.FONT_HERSHEY_SIMPLEX  # Het lettertype voor de tekst
            fontScale = 1  # De schaal van het lettertype
            color = (255, 0, 0)  # De kleur van de tekst (blauw)
            thickness = 2  # De dikte van de tekstlijnen
            
            # Voeg de klassenaam en object details toe aan het beeld
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)

    # Stop de loop af als de 'q'-toets wordt ingedrukt
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()