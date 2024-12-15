
# PPE Detection Project

This project focuses on detecting Personal Protective Equipment (PPE) such as helmets, vests, hairnets, and masks in images and videos. 
The project integrates various approaches including YOLO, MediaPipe, and GroundingDINO, with functionalities like video tracking and temporal smoothing.

## Features
- **YOLO-based detection**:
  - Trained on a merged dataset from Roboflow.
  - Supports detection in both images and videos.
  - Includes ByteTrack for video object tracking and temporal smoothing.
- **Alternative methods**:
  - **MediaPipe**: Used for simpler detection tasks, though less effective.
  - **GroundingDINO**: Provides another approach to detection but with room for improvement.

## Project Structure
- `train_n_video_test.ipynb`: Jupyter notebook containing the main workflow (training, testing, and tracking).
- `data/weights/`: Pre-trained YOLO models for different classes.
- `groundingdino/`: Scripts for GroundingDINO-based detection and training.
- `mediapipe&dlib/`: Scripts for MediaPipe and dlib-based detection.
- `data/`: Dataset preparation scripts and utilities.

## Getting Started
### Prerequisites
Make sure you have Python 3.8 or higher installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Usage
1. Train the YOLO model using `train_n_video_test.ipynb`.
2. Test the detection on images and videos with pre-trained weights.
3. Use the tracking functionality (ByteTrack) for video analysis.

### Example
Run the notebook for YOLO detection and tracking:
```bash
jupyter notebook train_n_video_test.ipynb
```

## Model Weights
Pre-trained weights are included in the `data/weights/` directory for different classes (helmet, vest, hairnet, and mask).

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

For further assistance, feel free to open an issue or contact the author.
