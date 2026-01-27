import torch # Import torch first to ensure CUDA loads correctly
import cv2
import numpy as np
import time
import os
import requests
import argparse
import csv
from collections import deque
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from emotion_recognizer import HSEmotionRecognizer

# --- Configuration ---
YOLO_WEIGHTS_URL = "https://huggingface.co/Bingsu/yolov8-face/resolve/main/yolov8n-face.pt"
YOLO_WEIGHTS_PATH = "yolov8n-face.pt"
CONF_THRESHOLD = 0.5
FACE_SIZE = 160 # InceptionResnetV1 requires 160x160
MICRO_EXPRESSION_THRESHOLD = 0.35 # Probability jump to trigger alert (0.0-1.0)


def download_weights(url, path):
    """Downloads the model weights if they don't exist."""
    if not os.path.exists(path):
        print(f"Weights not found at {path}. Downloading...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading weights: {e}")
            print("Please download manually or check your internet connection.")
            exit(1)
    else:
        print(f"Weights found at {path}.")

def get_args():
    parser = argparse.ArgumentParser(description="Real-time Face Feature Extraction Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam or path to video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save CSV output (e.g., features.csv)")
    parser.add_argument("--view", action="store_true", help="Visualize processing even when saving to file")
    return parser.parse_args()

def main():
    args = get_args()
    
    # --- 1. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type != 'cuda':
        print("WARNING: CUDA is not available. Running on CPU will be slow.")

    # --- 2. Load Models ---
    download_weights(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_PATH)
    print("Loading models...")
    try:
        yolo_model = YOLO(YOLO_WEIGHTS_PATH)
        yolo_model.to(device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        # Initialize Emotion Recognizer
        # model_name options: 'enet_b0_8_best_vgaf', 'enet_b0_8_best_afew'
        fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # --- 3. Initialize Source ---
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source '{source}'.")
        return

    # --- 4. Prepare Output CSV ---
    csv_file = None
    csv_writer = None
    
    if args.output:
        print(f"Saving features to: {args.output}")
        csv_file = open(args.output, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        # Header: Frame_ID, Timestamp, Feature_0 ... Feature_511
        # Header: Frame_ID, Timestamp, Emotion, Feature_0 ... Feature_511
        header = ['Frame_ID', 'Timestamp', 'Emotion'] + [f'Feat_{i}' for i in range(512)]
        csv_writer.writerow(header)

    print("Starting pipeline... Press 'q' to quit.")
    
    frame_id = 0
    prev_time = 0
    
    # Temporal Buffer for LSTM (stores last 30 frames of features)
    feature_buffer = deque(maxlen=30)
    # Score Buffer for Micro-Expression Spotting (stores last 30 frames of emotion probabilities)
    score_buffer = deque(maxlen=15) # Shorter buffer for baseline calculation
    
    # Micro-expression state
    micro_alert_until = 0
    micro_label = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or failed to grab frame.")
            break

        frame_id += 1
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        # --- A. Face Detection (YOLO) ---
        results = yolo_model(frame, verbose=False, conf=CONF_THRESHOLD)
        
        # Strategy: Find Largest Face if saving to file (to avoid noise)
        best_face = None
        max_area = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x1 >= x2 or y1 >= y2:
                    continue
                
                area = (x2 - x1) * (y2 - y1)
                
                # If we are processing a video file, we prioritize the largest face
                if args.output:
                    if area > max_area:
                        max_area = area
                        best_face = (x1, y1, x2, y2)
                else:
                    # Webcam mode: process all faces immediately (visual demo)
                    micro_dat = process_face(frame, (x1, y1, x2, y2), resnet, fer, device, frame_id, current_time, csv_writer=None, buffer=None, score_buffer=score_buffer)
                    
                    # Handle Alert persistence
                    if micro_dat:
                        micro_alert_until = time.time() + 1.0 # Display for 1 second
                        micro_label = micro_dat
        
        # Display persistent alert if active
        if time.time() < micro_alert_until:
             cv2.putText(frame, f"MICRO: {micro_label}!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
             cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

        # If writing to CSV, only process the single best face found
        if args.output and best_face:
            process_face(frame, best_face, resnet, fer, device, frame_id, current_time, csv_writer, buffer=feature_buffer, score_buffer=score_buffer)

        # --- Display FPS & Show (Optional) ---
        # Show window if it's webcam OR if --view flag is set
        if isinstance(source, int) or args.view:
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Real-time Feature Extraction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Progress indicator for file processing without window
            if frame_id % 30 == 0:
                print(f"Processing Frame {frame_id}...", end='\r')

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()
    print("\nPipeline stopped.")

def process_face(frame, box, resnet, fer, device, frame_id, timestamp, csv_writer=None, buffer=None, score_buffer=None):
    x1, y1, x2, y2 = box
    
    # Crop
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0: return

    # Resize & Preprocess
    face_resized = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).float()
    face_tensor = (face_tensor - 127.5) / 128.0
    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    # Inference (FaceNet)
    with torch.no_grad():
        embedding = resnet(face_tensor)
    
    embedding_np = embedding.cpu().numpy().flatten()
    
    # Emotion Recognition
    emotion, scores = fer.predict_emotions(face_rgb, logits=False)
    
    # Micro-Expression Spotting Logic
    micro_detected = None
    if score_buffer is not None:
        if len(score_buffer) > 5: # Ensure enough frames for a baseline
            # Calculate baseline (avg of last 5 frames)
            baseline = np.mean(list(score_buffer)[-5:], axis=0) # Use only the last 5 for baseline
            # Difference (Current - Baseline)
            diff = scores - baseline
            # Find max jump
            max_diff_idx = np.argmax(diff)
            max_diff_val = diff[max_diff_idx]
            
            if max_diff_val > MICRO_EXPRESSION_THRESHOLD:
                micro_emotion = fer.idx_to_class[max_diff_idx]
                micro_detected = micro_emotion
                print(f"!!! MICRO-EXPRESSION DETECTED: {micro_emotion} (Intensity: {max_diff_val:.2f}) !!!")

        score_buffer.append(scores)

    # Update Buffer
    if buffer is not None:
        buffer.append(embedding_np)
        # Note: In a real LSTM usage, you would pass 'list(buffer)' to the LSTM here.

    # Save to CSV
    if csv_writer:
        row = [frame_id, timestamp, emotion] + embedding_np.tolist()
        csv_writer.writerow(row)
    else:
        # Visual Print for Webcam
        print(f"Emotion: {emotion} | Vector: {embedding_np[:5]}...")
        
        # Draw Box & Emotion
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return micro_detected

if __name__ == "__main__":
    main()
