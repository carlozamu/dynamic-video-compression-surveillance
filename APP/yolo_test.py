from ultralytics import YOLO
import cv2

# Percorso del modello e dell'immagine di test
yolo_model = "yolov8s.pt"
test_image_path = "C:/Users/gianl/Pictures/wallpapers/IMG_9991.jpg"  # Sostituisci con un'immagine di prova

# Carica il modello
model = YOLO(yolo_model)

# Leggi l'immagine
img = cv2.imread(test_image_path)

# Effettua una previsione
results = model.predict(img, save=False, conf=0.25)

# Stampa i risultati
print("Risultati YOLO:", results)
for box in results[0].boxes.data.tolist():
    print(f"Box: {box}")
