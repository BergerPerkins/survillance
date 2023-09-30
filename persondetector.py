import cv2
import torch
import numpy as np

# Load the YOLOv5 model with pre-trained weights (you need to specify the path)
model = torch.hub.load('ultralytics/yolov5:v5.0', 'yolov5s', pretrained=True)

# Set the device to 'cuda' for GPU inference if available, else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the model to evaluation mode
model.eval()

# Define the class index for persons (YOLOv5 usually labels persons as 0)
person_class_index = 0

# Load an image for detection (you can replace this with your image path)
image = cv2.imread('C:/Users/Berger/Downloads/download_s.jpg')

# Preprocess the image for inference
img = model.preprocess(image)

# Perform inference
with torch.no_grad():
    detections = model(img.to(device))[0]  # Get detections for the first image in the batch

# Filter detections to keep only persons
persons = detections[detections[:, -1] == person_class_index]

# Loop through the detected persons and draw bounding boxes
for person in persons:
    x1, y1, x2, y2, confidence = person[:5]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

# Save or display the image with bounding boxes
cv2.imshow('Person Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
