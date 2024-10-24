import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO('C:\\your\\url\\model.pt')

# Read image with Pillow
image = Image.open('C:\\your\\url\\photo.png')

# Convert image to YOLO compatible format
image = image.convert("RGB")

# Run detection with YOLOv8
results = model(image)

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Extract coordinates of the detected boxes and apply OCR
for result in results:
    boxes = result.boxes.xyxy.numpy()  
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Imprimir las coordenadas de las cajas
        print(f"Coordenadas de la caja: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        cropped_image = image.crop((x1, y1, x2, y2))  
        cropped_image_np = np.array(cropped_image)  
        ocr_result = reader.readtext(cropped_image_np) # Extract text
        
        # Print only detected text
        for res in ocr_result:
            print(res[1])  # res[1] contains the extracted text
