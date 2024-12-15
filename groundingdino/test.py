import os
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from ultralytics import YOLO
import cv2
from PIL import Image

# Accessory classification model setup
data_folder = 'C:/Users/ze/Desktop/metasmart2/organized_dataset/'
accessory_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = ImageFolder(root=data_folder, transform=accessory_transform)

# Define and load the accessory detection model
from torchvision.models import efficientnet_b0
model = efficientnet_b0(weights='IMAGENET1K_V1')  # Use EfficientNet as in training
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(dataset.classes))  # Replace final layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load('accessory_model.pth', map_location=device))
model.eval()

# Load YOLOv8 model for human detection
yolo_model = YOLO("yolov8n.pt")  # Use YOLOv8 nano for fast human detection

# Open video capture
video_path = 'input_video.mp4'
output_video_path = 'output_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Human detection using YOLOv8
    results = yolo_model(frame)
    for detection in results[0].boxes:
        # Get bounding box coordinates and class
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        confidence = detection.conf.item()
        class_id = int(detection.cls.item())

        # Check if the detection is for a "person"
        if class_id == 0 and confidence > 0.5:
            # Draw the bounding box around the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Crop the region for accessory detection
            human_roi = frame[y1:y2, x1:x2]

            try:
                human_img = Image.fromarray(cv2.cvtColor(human_roi, cv2.COLOR_BGR2RGB))
                input_tensor = accessory_transform(human_img).unsqueeze(0).to(device)

                # Accessory classification
                with torch.no_grad():
                    output = model(input_tensor)
                    _, pred = output.max(1)
                    label = dataset.classes[pred.item()]

                # Draw a separate bounding box and label for the accessory detection
                cv2.rectangle(frame, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing ROI: {e}")

    # Write frame to output video
    out.write(frame)
# Release resources
cap.release()
out.release()
