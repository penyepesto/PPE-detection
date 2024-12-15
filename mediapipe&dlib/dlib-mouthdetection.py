import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from dlib's repository

# Function to annotate the video frame
def annotate_frame(frame, detection_status):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for feature, status, position in detection_status:
        color = (0, 0, 255) if status else (0, 255, 0)  # Red if detected, Green otherwise
        cv2.putText(frame, feature, position, font, 0.7, color, 2, cv2.LINE_AA)

    return frame

# Process the video
def process_video(input_video_path, output_video_path):
    # Load input video
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        detection_status = []
        if faces:  # Ensure landmarks are calculated only if a face is detected
            for face in faces:
                # Get landmarks
                landmarks = predictor(gray, face)

                # Extract key features (mouth, nose, chin)
                mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
                nose = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 36)]
                chin = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(6, 11)]

                # Draw key points
                for point in mouth + nose + chin:
                    cv2.circle(frame, point, 2, (255, 0, 0), -1)

                # Feature detection status
                detection_status.append(("Mouth", bool(mouth), (50, 50)))
                detection_status.append(("Nose", bool(nose), (50, 100)))
                detection_status.append(("Chin", bool(chin), (50, 150)))
        else:
            # If no face detected, set features as not detected
            detection_status.append(("Mouth", False, (50, 50)))
            detection_status.append(("Nose", False, (50, 100)))
            detection_status.append(("Chin", False, (50, 150)))

        # Annotate frame
        frame = annotate_frame(frame, detection_status)

        # Write the frame into the output video
        out.write(frame)

        # Display the result (optional)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Input and output video paths
input_video_path = "input_video.mp4"  # Replace with your input video path
output_video_path = "output_video.mp4"

# Process the video
process_video(input_video_path, output_video_path)
