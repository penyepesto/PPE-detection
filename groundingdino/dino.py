import os
import torch
from PIL import Image
import cv2
from groundingdino.util.inference import load_model, predict
from torchvision.transforms import functional as F
from transformers import CLIPProcessor, CLIPModel

# Load Grounding DINO model
def load_grounding_dino():
    config_path = "C:/Users/ze/Desktop/metasmart2/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    model_path = "C:/Users/ze/Desktop/metasmart2/groundingdino_swinb_cogcoor.pth"
    model = load_model(config_path, model_path)
    return model

# Load CLIP for embedding generation
def load_clip_model():
    clip_model = CLIPModel.from_pretrained("C:/Users/ze/Desktop/metasmart2/clip-vit-base-patch16")
    clip_processor = CLIPProcessor.from_pretrained("C:/Users/ze/Desktop/metasmart2/clip-vit-base-patch16")
    return clip_model, clip_processor

# Encode reference images from dataset
def encode_reference_images(dataset_path, clip_model, clip_processor):
    reference_features = []
    labels = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
            reference_features.append(features)
            labels.append(class_name)  # Class name from folder
    return torch.cat(reference_features), labels

# Match detected features with reference features
def match_features(detected_features, reference_features):
    similarities = torch.matmul(detected_features, reference_features.T)
    best_match = similarities.argmax(dim=1)
    return best_match
def detect_objects_in_video(video_path, dataset_path, output_video_path):
    # Load models
    grounding_dino = load_grounding_dino()
    clip_model, clip_processor = load_clip_model()

    # Encode reference images
    reference_features, labels = encode_reference_images(dataset_path, clip_model, clip_processor)

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Caption used for Grounding DINO
    caption = ", ".join(labels)  # Combine all class names into a single caption
    box_threshold = 0.3  # Adjust as needed for box confidence
    text_threshold = 0.25  # Adjust as needed for text matching confidence
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for Grounding DINO
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert to tensor

        # Move the tensor to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)

        # Wrap the tensor in a list for Grounding DINO
        image_list = [image_tensor]

        # Predict objects in the frame
        caption = "object"  # Modify based on your dataset
        box_threshold = 0.3
        text_threshold = 0.25

        detections = predict(
            model=grounding_dino,
            image=image_list,  # List of tensors
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # Process detections (similar to previous steps)
        for i, bbox in enumerate(detections["boxes"]):
            x1, y1, x2, y2 = map(int, bbox)
            label = "Detected Object"  # Replace with your matching logic
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)



    cap.release()
    out.release()
    print("Detection completed. Output saved to:", output_video_path)


# Paths
dataset_path = "C:/Users/ze/Desktop/metasmart2/organized_dataset"
video_path = "C:/Users/ze/Desktop/metasmart2/input_video.mp4"
output_video_path = "C:/Users/ze/Desktop/metasmart2/output_video.mp4"

# Run detection
detect_objects_in_video(video_path, dataset_path, output_video_path)
