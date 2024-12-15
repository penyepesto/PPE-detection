import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from transformers import CLIPProcessor, CLIPModel
from groundingdino.util.inference import load_model, predict

device = torch.device( "cpu")

# Load person detection model (YOLO pre-trained on COCO)
person_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
person_detector.conf = 0.5
person_detector.classes = [0]  # Person class

# Load GroundingDINO model
def load_grounding_dino():
    config_path = "C:/Users/ze/Desktop/metasmart2/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    model_path = "C:/Users/ze/Desktop/metasmart2/groundingdino_swinb_cogcoor.pth"
    model = load_model(config_path, model_path)
    return model.to(device)

grounding_dino = load_grounding_dino()

# Load CLIP
clip_model = CLIPModel.from_pretrained("C:/Users/ze/Desktop/metasmart2/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("C:/Users/ze/Desktop/metasmart2/clip-vit-base-patch16")

# Prepare reference embeddings
def encode_reference_images(dataset_path, clip_model, clip_processor):
    reference_embeddings = {}
    for accessory_class in ["helmet", "hairnet", "mask", "vest"]:
        class_path = f"{dataset_path}/{accessory_class}"
        emb_list = []
        for img_name in ["front.png", "right.png", "left.png"]:
            img_path = f"{class_path}/{img_name}"
            image = Image.open(img_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb_list.append(emb)
        # Average embeddings or just keep them all
        reference_embeddings[accessory_class] = torch.mean(torch.stack(emb_list), dim=0)
    return reference_embeddings

reference_embeddings = encode_reference_images("C:/Users/ze/Desktop/metasmart2/organized_dataset", clip_model, clip_processor)

# Thresholds and smoothing settings
SIMILARITY_THRESHOLD = 0.3
HOLD_FRAMES = 10

# State tracking for person
person_states = {}  # person_id -> {accessory: {'state': bool, 'count': int}}

def update_accessory_states(person_id, detected_accessories):
    if person_id not in person_states:
        person_states[person_id] = {
            'helmet': {'state': False, 'count': 0},
            'hairnet': {'state': False, 'count': 0},
            'mask': {'state': False, 'count': 0},
            'vest': {'state': False, 'count': 0}
        }
    for acc in person_states[person_id]:
        if acc in detected_accessories:
            # Detected accessory this frame
            person_states[person_id][acc]['count'] = min(person_states[person_id][acc]['count'] + 1, HOLD_FRAMES)
        else:
            # Not detected
            person_states[person_id][acc]['count'] = max(person_states[person_id][acc]['count'] - 1, 0)

        # Determine final state: accessory is "in use" if count > half the hold_frames
        person_states[person_id][acc]['state'] = (person_states[person_id][acc]['count'] > HOLD_FRAMES // 2)
def detect_accessories_with_groundingdino_and_clip(person_crop):
    # Convert to PIL and then to tensor (C,H,W)
    pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
    image_tensor = F.to_tensor(pil_image).to(device)  # No unsqueeze(0)

    detected = []
    for acc in ["helmet", "hairnet", "mask", "vest"]:
        # Set your prompt
        if acc == "helmet":
            prompt = "a person wearing a helmet"
        elif acc == "hairnet":
            prompt = "a person wearing a hairnet"
        elif acc == "mask":
            prompt = "a person wearing a face mask"
        elif acc == "vest":
            prompt = "a person wearing a safety vest"

        # Pass image_tensor directly; `predict` adds batch dimension internally
        detections = predict(
            model=grounding_dino,
            image=image_tensor,   
            caption=prompt,
            box_threshold=0.3,
            text_threshold=0.25
        )

        # If GroundingDINO finds boxes, verify with CLIP
        if len(detections["boxes"]) > 0:
            x1, y1, x2, y2 = map(int, detections["boxes"][0])
            crop = pil_image.crop((x1, y1, x2, y2))
            inputs = clip_processor(images=crop, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            sim = torch.matmul(emb, reference_embeddings[acc].T)
            if sim.item() > SIMILARITY_THRESHOLD:
                detected.append(acc)

    return detected


def main():
    video_path = "C:/Users/ze/Desktop/metasmart2/input_video.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter("C:/Users/ze/Desktop/metasmart2/output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    frame_count = 0

    person_id = 0  # if there's only one person, you can just assume person_id = 0
                   # if multiple persons, you might need a tracker, but let's keep it simple.

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Person detection
        results = person_detector(frame)
        # Assuming we have one main person, choose the highest confidence person
        persons = [x for x in results.xyxy[0].cpu().numpy() if int(x[5]) == 0]
        if len(persons) > 0:
            # Sort by confidence descending
            persons = sorted(persons, key=lambda x:x[4], reverse=True)
            x1, y1, x2, y2, conf, cls = persons[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            person_crop = frame[y1:y2, x1:x2]

            # Detect accessories
            detected_accessories = detect_accessories_with_groundingdino_and_clip(person_crop)
            update_accessory_states(person_id, detected_accessories)

            # Draw results
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            # Show state
            y_text = y1 - 10
            for acc in ["helmet", "hairnet", "mask", "vest"]:
                state = "using" if person_states[person_id][acc]['state'] else "not using"
                cv2.putText(frame, f"{acc}: {state}", (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                y_text -= 15

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print("Done!")

if __name__ == "__main__":
    main()
