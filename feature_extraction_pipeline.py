import torch # Import torch first to ensure CUDA loads correctly
import cv2
import numpy as np
import time
import os
import requests
import argparse
import csv
from collections import deque, defaultdict
from typing import Optional, List, Tuple, Deque, Any, Union

# --- MONKEYPATCH: Fix WinError 1337 (Invalid Security ID) in ultralytics/pathlib ---
import pathlib
_orig_exists = pathlib.Path.exists
def _safe_exists(self):
    try:
        return _orig_exists(self)
    except OSError as e:
        # Catch WinError 1337 (Security ID structure is invalid) which happens on protected drives/folders
        if getattr(e, 'winerror', 0) == 1337:
             return False
        raise
    except Exception as e:
        if "1337" in str(e): # Fallback for other exception types wrapping the error
             return False
        raise
pathlib.Path.exists = _safe_exists
# -----------------------------------------------------------------------------------

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from emotion_recognizer import HSEmotionRecognizer
from lstm_model import MicroExpressionLSTM
from train_lstm import MODEL_SAVE_PATH, INPUT_SIZE

# --- Configuration ---
YOLO_WEIGHTS_URL = "https://huggingface.co/Bingsu/yolov8-face/resolve/main/yolov8n-face.pt"
YOLO_WEIGHTS_PATH = "yolov8n-face.pt"
CONF_THRESHOLD = 0.5
FACE_SIZE = 160 # InceptionResnetV1 requires 160x160
MICRO_EXPRESSION_THRESHOLD = 0.35 # Probability jump to trigger alert (0.0-1.0)
EMOTION_MAP = {
    'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 
    'fear': 4, 'sadness': 5, 'others': 6
}

class ProbabilitySmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed_probs = None
        
    def update(self, current_probs):
        if self.smoothed_probs is None:
            self.smoothed_probs = current_probs
        else:
            self.smoothed_probs = self.alpha * current_probs + (1 - self.alpha) * self.smoothed_probs
        return self.smoothed_probs

class AlertCooldown:
    def __init__(self, cooldown_time=2.0):
        self.cooldown_time = cooldown_time
        self.last_alert_time = 0
        
    def can_alert(self):
        return (time.time() - self.last_alert_time) > self.cooldown_time
        
    def trigger(self):
        self.last_alert_time = time.time()

def download_weights(url: str, path: str) -> None:
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

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time Face Feature Extraction Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam or path to video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save CSV output (e.g., features.csv)")
    parser.add_argument("--view", action="store_true", help="Visualize processing even when saving to file")
    parser.add_argument("--duration", type=float, default=0, help="Duration to analyze in seconds (0 for infinite)")
    return parser.parse_args()

def setup_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type != 'cuda':
        print("WARNING: CUDA is not available. Running on CPU will be slow.")
    return device

def load_models(device: torch.device) -> Tuple[YOLO, InceptionResnetV1, Optional[HSEmotionRecognizer], Optional[MicroExpressionLSTM]]:
    download_weights(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_PATH)
    print("Loading models...")
    try:
        yolo_model = YOLO(YOLO_WEIGHTS_PATH)
        yolo_model.to(device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        try:
            fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
        except Exception as e:
            print(f"WARNING: Emotion Recognizer failed to load: {e}")
            fer = None

        lstm_model = None
        lstm_weights = MODEL_SAVE_PATH
        if os.path.exists(lstm_weights):
            print(f"Loading LSTM model from {lstm_weights}...")
            # Use imported INPUT_SIZE to match the model architecture (520 or 8)
            lstm_model = MicroExpressionLSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, num_classes=len(EMOTION_MAP))
            lstm_model.load_state_dict(torch.load(lstm_weights, map_location=device))
            lstm_model.to(device)
            lstm_model.eval()
            print("LSTM model loaded.")
        else:
            print(f"No LSTM weights found ({lstm_weights}). Using Heuristic Spotter.")
            
        print("Models loaded successfully.")
        return yolo_model, resnet, fer, lstm_model
    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)

def process_face(
    frame: np.ndarray, 
    box: Tuple[int, int, int, int], 
    resnet: InceptionResnetV1, 
    fer: Optional[HSEmotionRecognizer], 
    lstm_model: Optional[MicroExpressionLSTM], 
    device: torch.device, 
    frame_id: int, 
    timestamp: float, 
    csv_writer: Any, 
    score_buffer: Optional[Deque[np.ndarray]],
    smoother: Optional[ProbabilitySmoother] = None,
    cooldown: Optional[AlertCooldown] = None,
    draw: bool = False
) -> Tuple[Optional[str], float, str]:
    
    x1, y1, x2, y2 = box
    
    # Crop
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0: return None, 0.0

    # Resize & Preprocess
    face_resized = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Prepare Tensor for FaceNet
    face_tensor = torch.from_numpy(face_rgb).float()
    face_tensor = (face_tensor - 127.5) / 128.0
    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # FaceNet Inference
    with torch.no_grad():
        embedding = resnet(face_tensor)
    embedding_np = embedding.cpu().numpy().flatten()
    
    # Emotion Recognition
    emotion = "Unknown"
    scores = np.zeros(8)
    if fer is not None:
        emotion, scores = fer.predict_emotions(face_rgb, logits=False)
    
    # Logic for Micro-Expression Spotting
    micro_detected = None
    confidence = 0.0
    
    if score_buffer is not None:
        # Create combined feature: [512 Feature, 8 Scores]
        combined_feature = np.concatenate((embedding_np, scores))
        score_buffer.append(combined_feature)
        
        # --- A. LSTM Logic (Preferred) ---
        if lstm_model is not None and len(score_buffer) == score_buffer.maxlen:
            
            # Prepare sequence: (1, 30, 520) usually
            seq = np.array(list(score_buffer))
            
            # --- CRITICAL: MATCH INPUT SIZE ---
            # If the loaded model expects 8 inputs (Emotion Only) but we have 520 in buffer:
            if seq.shape[1] > INPUT_SIZE:
                # Assuming the last N features are the emotion scores
                seq = seq[:, -INPUT_SIZE:]
            
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output, attn_weights = lstm_model(seq_tensor)
                # Output is [Batch, 7] -> Multi-class scores
                probs = torch.softmax(output, dim=1)
                
                # Apply Smoothing
                current_probs = probs.cpu().numpy()[0]
                if smoother:
                    current_probs = smoother.update(current_probs)
                
                # Show top prediction
                pred_idx_val = np.argmax(current_probs)
                max_prob_val = current_probs[pred_idx_val]
                
                confidence = max_prob_val
                
                # Check Cooldown
                can_trigger = True
                if cooldown:
                    can_trigger = cooldown.can_alert()

                # Optimization: Increased threshold to 0.75 for stability
                if max_prob_val > 0.75 and can_trigger:
                    # Assuming we map index to emotion name if needed
                    keys = list(EMOTION_MAP.keys())
                    vals = list(EMOTION_MAP.values())
                    pred_name = keys[vals.index(pred_idx_val)] if pred_idx_val in vals else str(pred_idx_val)
                    
                    # Ignore "others"/neutral if it's the dominant class
                    # MACRO-SUPPRESSION: If the base emotion (HSEmotion) is already detecting the same emotion,
                    # e.g., "Happiness" -> "LSTM: Happiness", then it's not a micro-expression, it's just a normal one.
                    # Convert both to lowercase for comparison.
                    base_emotion_lower = emotion.lower() if emotion else ""
                    
                    if pred_name != 'others':
                        if pred_name in base_emotion_lower:
                             # It's already obvious, don't flag as micro
                             pass
                        else:
                            micro_detected = f"LSTM: {pred_name}"
                            if cooldown:
                                cooldown.trigger()
                    
                    # Return the dominant emotion even if no micro-alert
                    return micro_detected, confidence, pred_name

    # Save to CSV
    if csv_writer:
        row = [frame_id, timestamp, emotion] + embedding_np.tolist()
        csv_writer.writerow(row)
        
    # Draw (Decoupled from CSV)
    if draw:
        # Draw Box & Emotion
        color = (0, 255, 0)
        thickness = 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label with Confidence
        # Just show the emotion name (whether originating from LSTM or HSEmotion)
        # If micro_detected exists, it has the specific emotion name, use that but clean it.
        if micro_detected:
             # micro_detected is like "LSTM: happiness" -> just "happiness"
             display_emotion = micro_detected.replace("LSTM: ", "").title()
             label = f"{display_emotion} ({confidence:.0%})"
        else:
             display_emotion = emotion.title() if emotion else "Unknown"
             label = f"{display_emotion}"
            
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Ensure three return values
    fallback_ret = emotion if emotion else "Neutral"
    return micro_detected, confidence, fallback_ret

def main():
    args = get_args()
    device = setup_device()
    yolo_model, resnet, fer, lstm_model = load_models(device)
    
    # Initialize Source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source '{source}'.")
        return

    # Prepare Output CSV
    csv_file = None
    csv_writer = None
    if args.output:
        print(f"Saving features to: {args.output}")
        csv_file = open(args.output, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        # Header: Frame_ID, Timestamp, Emotion, Feature_0 ... Feature_511
        header = ['Frame_ID', 'Timestamp', 'Emotion'] + [f'Feat_{i}' for i in range(512)]
        csv_writer.writerow(header)

    print("Starting pipeline... Press 'q' to quit.")
    
    frame_id = 0
    prev_time = 0
    
    # Buffers
    # Buffer for LSTM (stores last 30 frames of combined features)
    score_buffer = deque(maxlen=30)
    
    # Alert Logic Helpers
    smoother = ProbabilitySmoother(alpha=0.3)
    cooldown = AlertCooldown(cooldown_time=1.5)
    
    micro_alert_until = 0
    micro_label = ""
    
    # Session Statistics
    emotion_counts = defaultdict(int) 
    total_processed_frames = 0
    start_time = time.time()
    
    try:
        while True:
            # Check duration
            elapsed = time.time() - start_time
            if args.duration > 0 and elapsed > args.duration:
                print(f"\nDuration limit ({args.duration}s) reached.")
                break

            ret, frame = cap.read()
            if not ret:
                print("End of stream or failed to grab frame.")
                break

            frame_id += 1
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            # Face Detection
            results = yolo_model(frame, verbose=False, conf=CONF_THRESHOLD)
            
            best_face = None
            max_area = 0

            # Find best face
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    h_img, w_img, _ = frame.shape
                    
                    # Clip coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    if x1 >= x2 or y1 >= y2:
                        continue
                        
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_face = (x1, y1, x2, y2)

            # Process Face
            if best_face:
                should_draw = isinstance(source, int) or args.view
                micro_text, confidence, current_emotion_name = process_face(
                    frame, best_face, resnet, fer, lstm_model, device, 
                    frame_id, current_time, csv_writer, score_buffer,
                    smoother=smoother, cooldown=cooldown, 
                    draw=should_draw
                )
                
                # Global Alert Logic for persistent display
                if micro_text:
                    micro_label = micro_text
                    micro_alert_until = current_time + 1.5 # Show for 1.5s
                    
                # [STATS] Accumulate emotion for summary
                if micro_text:
                     # Clean key for stats
                     clean_key = micro_text.replace("LSTM: ", "").title()
                     emotion_counts[clean_key] += 1
                else: 
                     # Log the actual detected emotion (e.g., "happiness", "neutral")
                     clean_key = current_emotion_name.title() if current_emotion_name else "Neutral"
                     emotion_counts[clean_key] += 1
                
                total_processed_frames += 1
            
            # Show window
            if isinstance(source, int) or args.view:
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Real-time Feature Extraction', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if frame_id % 30 == 0:
                    print(f"Processing Frame {frame_id}...", end='\r')
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()
        
        # Print Summary
        if total_processed_frames > 0:
            print("\n" + "="*40)
            print(f" SESSION SUMMARY ({time.time() - start_time:.1f}s)")
            print("="*40)
            print(f"Total Frames Analyzed: {total_processed_frames}")
            for emo, count in emotion_counts.items():
                pct = (count / total_processed_frames) * 100
                print(f"{emo:<20}: {pct:.1f}%")
            print("="*40 + "\n")
            
        print("\nPipeline stopped.")

if __name__ == "__main__":
    main()
