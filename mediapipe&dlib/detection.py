import os
import cv2
import torch
import numpy as np
import logging
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
import mediapipe as mp
import torchvision.transforms as transforms
from PIL import Image

logging.basicConfig(level=logging.INFO)

# ----------------------------
# Configuration
# ----------------------------
VIDEO_PATH = "input_video.mp4"
OUTPUT_PATH = "output_video2.mp4"
PERSON_CONF_THRESHOLD = 0.5
HOLD_FRAMES = 30

# ----------------------------
# Model: Person Detection (YOLOv8)
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
person_model = YOLO('yolov8n.pt').to(device)
person_model.overrides['conf'] = PERSON_CONF_THRESHOLD
person_model.overrides['classes'] = [0]  # Person class only

# ----------------------------
# ByteTrack Setup
# ----------------------------
class ByteTrackArgs:
    def __init__(self):
        self.track_thresh = 0.6
        self.match_thresh = 0.8
        self.track_buffer = 30
        self.mot20 = False

tracker_args = ByteTrackArgs()
tracker = BYTETracker(tracker_args, frame_rate=30)

# ----------------------------
# Mediapipe Setup
# ----------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ----------------------------
# Mask Classification Model
# ----------------------------
# Assume we have a model that outputs:
# 0 -> mask
# 1 -> no_mask
mask_model = torch.load("face_mask_detector.pth", map_location=device)
mask_model.eval()

# Transform to apply to the face image before classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# State Tracking
# ----------------------------
person_states = {}
# person_states[person_id] = {
#   'mask': {'state': False, 'hold_counter': 0}
# }

def init_person_state(person_id):
    if person_id not in person_states:
        person_states[person_id] = {
            'mask': {'state': False, 'hold_counter': 0}
        }

def update_mask_state(person_id, detected_mask, hold_frames=HOLD_FRAMES):
    init_person_state(person_id)
    state_info = person_states[person_id]['mask']

    if detected_mask:
        state_info['hold_counter'] = hold_frames
        state_info['state'] = True
    else:
        if state_info['hold_counter'] > 0:
            state_info['hold_counter'] -= 1
        else:
            state_info['state'] = False

def classify_mask(face_crop):
    # face_crop is a BGR image
    pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    img_t = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = mask_model(img_t)
        pred = torch.argmax(logits, dim=1).item()

    # If pred == 0: mask, if pred == 1: no_mask
    return (pred == 0)

def get_face_bbox(image, detection):
    h, w, _ = image.shape
    bbox = detection.location_data.relative_bounding_box
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = x1 + int(bbox.width * w)
    y2 = y1 + int(bbox.height * h)
    return x1, y1, x2, y2

def process_frame(frame, frame_width, frame_height):
    # Person Detection
    results = person_model.predict(frame, verbose=False)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:  # person
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append([xmin, ymin, xmax, ymax, conf])

    if len(detections) == 0:
        return frame, []

    detections = np.array(detections)
    online_tracks = tracker.update(detections, [frame_width, frame_height], [frame_width, frame_height])

    for track in online_tracks:
        if not track.is_activated:
            continue

        track_id = track.track_id
        xmin, ymin, xmax, ymax = map(int, track.tlbr)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(frame_width, xmax), min(frame_height, ymax)

        person_crop = frame[ymin:ymax, xmin:xmax]

        # Face Detection using Mediapipe
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(person_rgb)

        detected_mask = False
        if face_results.detections:
            detection = face_results.detections[0]
            fx1, fy1, fx2, fy2 = get_face_bbox(person_crop, detection)
            # Adjust to full-frame coords
            fx1 += xmin
            fy1 += ymin
            fx2 += xmin
            fy2 += ymin

            # Crop face from original frame
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(frame_width, fx2), min(frame_height, fy2)

            face_crop = frame[fy1:fy2, fx1:fx2]
            if face_crop.size > 0:
                # Classify mask
                detected_mask = classify_mask(face_crop)

        # Update mask state
        update_mask_state(track_id, detected_mask)

        # Draw bounding box and mask label
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        mask_state = person_states[track_id]['mask']['state']
        color = (0, 255, 0) if mask_state else (0, 0, 255)
        label = f"mask: {'using' if mask_state else 'removed'}"
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, online_tracks

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, _ = process_frame(frame, frame_width, frame_height)
        out.write(processed_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            logging.info(f"Processed {frame_count} frames.")

    cap.release()
    out.release()
    print(f"Processed video saved at {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    main()
