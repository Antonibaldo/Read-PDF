# PDF Reader Project: Text Detection and Extraction with YOLOv8 and EasyOCR
This project integrates YOLOv8 and EasyOCR to detect and extract text from images, such as invoices or delivery notes.

## Key Features:
YOLOv8 detects text areas in the image by drawing bounding boxes around them.
EasyOCR extracts the text from each detected box.
## Workflow:
Load an image.
YOLOv8 detects text regions.
EasyOCR extracts text from these regions.
The result is a list of the detected texts along with their bounding box coordinates.
## Project Structure:
Pretrained YOLOv8 for text detection.
EasyOCR for multilingual text extraction.
Python script managing detection and extraction processes.
## Use Cases:
Automatic reading of documents like invoices and receipts.
Text extraction from scanned documents for automated analysis.
