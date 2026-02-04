import torch
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
        if getattr(e, 'winerror', 0) == 1337:
             return False
        raise
    except Exception as e:
        if "1337" in str(e):
             return False
        raise
pathlib.Path.exists = _safe_exists
# -----------------------------------------------------------------------------------

from ultralytics import YOLO
from emotion_recognizer import HSEmotionRecognizer
from lstm_model import MicroExpressionLSTM
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# --- Configuration ---
YOLO_WEIGHTS_URL = "https://huggingface.co/Bingsu/yolov8-face/resolve/main/yolov8n-face.pt"
YOLO_WEIGHTS_PATH = "yolov8n-face.pt"
CONF_THRESHOLD = 0.5
FACE_SIZE = 224 # ResNet18 uses 224x224
MICRO_EXPRESSION_THRESHOLD = 0.35
EMOTION_MAP = {
    'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 
    'fear': 4, 'sadness': 5, 'others': 6
}
INPUT_SIZE = 512 + 8
MODEL_SAVE_PATH = "lstm_celeba.pth"
BACKBONE_PATH = "celeba_backbone.pth"

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
            exit(1)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time Face Feature Extraction Pipeline (CelebA Experiment)")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam or path to video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save CSV output (e.g., features.csv)")
    parser.add_argument("--view", action="store_true", help="Visualize processing even when saving to file")
    parser.add_argument("--duration", type=float, default=0, help="Duration to analyze in seconds (0 for infinite)")
    return parser.parse_args()

def setup_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_models(device: torch.device):
    download_weights(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_PATH)
    print("Loading models...")
    try:
        yolo_model = YOLO(YOLO_WEIGHTS_PATH)
        yolo_model.to(device)
        
        # --- CelebA ResNet18 Backbone ---
        print(f"Loading CelebA Backbone from {BACKBONE_PATH}...")
        resnet = models.resnet18(pretrained=False)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 40)
        resnet.load_state_dict(torch.load(BACKBONE_PATH, map_location=device))
        
        # Remove head
        modules = list(resnet.children())[:-1]
        resnet = nn.Sequential(*modules)
        resnet.eval().to(device)
        # --------------------------------
        
        try:
            fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
        except Exception as e:
            print(f"WARNING: Emotion Recognizer failed to load: {e}")
            fer = None

        lstm_model = None
        if os.path.exists(MODEL_SAVE_PATH):
            print(f"Loading LSTM model from {MODEL_SAVE_PATH}...")
            lstm_model = MicroExpressionLSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, num_classes=len(EMOTION_MAP))
            lstm_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            lstm_model.to(device)
            lstm_model.eval()
            print("LSTM model loaded.")
        else:
            print(f"No LSTM weights found ({MODEL_SAVE_PATH}).")
            
        print("Models loaded successfully.")
        return yolo_model, resnet, fer, lstm_model
    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)

def process_face(
    frame: np.ndarray, 
    box: Tuple[int, int, int, int], 
    resnet: nn.Module, 
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
    if face_crop.size == 0: return None, 0.0, ""

    # Resize for ResNet18 (224x224)
    face_resized = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Prepare Tensor for ResNet (ImageNet Norm)
    face_tensor = torch.from_numpy(face_rgb).float() / 255.0
    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    face_tensor = face_tensor.permute(2, 0, 1) # C H W
    face_tensor = norm_transform(face_tensor).unsqueeze(0).to(device)

    # ResNet Inference
    with torch.no_grad():
        embedding = resnet(face_tensor) # [1, 512, 1, 1]
    embedding_np = embedding.view(-1).cpu().numpy() # [512]
    
    # Emotion Recognition
    emotion = "Unknown"
    scores = np.zeros(8)
    if fer is not None:
        # HSEmotion uses its own preprocessing
        emotion, scores = fer.predict_emotions(face_rgb, logits=False)
        
        # Bias Correction
        pred_idx = np.argmax(scores)
        if pred_idx == 0 and scores[0] < 0.60: emotion = 'Neutral'
        elif pred_idx == 1 and scores[1] < 0.60: emotion = 'Neutral'
    
    # Logic for Micro-Expression Spotting
    micro_detected = None
    confidence = 0.0
    
    if score_buffer is not None:
        combined_feature = np.concatenate((embedding_np, scores))
        score_buffer.append(combined_feature)
        
        if lstm_model is not None and len(score_buffer) == score_buffer.maxlen:
            seq = np.array(list(score_buffer)) # [30, 520]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output, attn_weights = lstm_model(seq_tensor)
                probs = torch.softmax(output, dim=1)
                
                current_probs = probs.cpu().numpy()[0]
                if smoother:
                    current_probs = smoother.update(current_probs)
                
                pred_idx_val = np.argmax(current_probs)
                max_prob_val = current_probs[pred_idx_val]
                confidence = max_prob_val
                
                can_trigger = True
                if cooldown: can_trigger = cooldown.can_alert()

                if max_prob_val > 0.75 and can_trigger:
                    keys = list(EMOTION_MAP.keys())
                    vals = list(EMOTION_MAP.values())
                    pred_name = keys[vals.index(pred_idx_val)] if pred_idx_val in vals else str(pred_idx_val)
                    
                    base_emotion_lower = emotion.lower() if emotion else ""
                    if pred_name != 'others':
                        if pred_name in base_emotion_lower:
                             pass
                        else:
                            micro_detected = f"LSTM: {pred_name}"
                            if cooldown: cooldown.trigger()
                    
                    if not micro_detected:
                        return None, confidence, pred_name

    # Draw
    if draw:
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if micro_detected:
             display_emotion = micro_detected.replace("LSTM: ", "").title()
             label = f"{display_emotion} ({confidence:.0%})"
        else:
             display_emotion = emotion.title() if emotion else "Unknown"
             label = f"{display_emotion}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return micro_detected, confidence, (emotion if emotion else "Neutral")

def main():
    args = get_args()
    device = setup_device()
    yolo_model, resnet, fer, lstm_model = load_models(device)
    
    source = args.source
    if source == "0":
        print("Select Mode:")
        print("1. Live Camera Mode")
        print("2. Video File Mode")
        try:
            mode = input("Enter mode (1 or 2) [Default: 1]: ").strip()
        except EOFError:
            mode = "1"

        if mode == "2":
            print("\n--- Video File Mode ---")
            video_path = input("Enter path to video file: ").strip()
            video_path = video_path.strip('"').strip("'")
            if not os.path.exists(video_path):
                print(f"Error: File not found at {video_path}")
                return
            source = video_path
            print(f"Processing video: {source}")
        else:
            print("\n--- Live Camera Mode ---")
            source = 0
            print("Using Webcam (Index 0)")
    else:
        if source.isdigit():
            source = int(source)
            print(f"Using Camera Index: {source}")
        else:
            print(f"Using Video File: {source}")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source '{source}'.")
        return

    csv_file = None
    csv_writer = None
    if args.output:
        print(f"Saving features to: {args.output}")
        csv_file = open(args.output, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        header = ['Frame_ID', 'Timestamp', 'Emotion'] + [f'Feat_{i}' for i in range(512)]
        csv_writer.writerow(header)

    print("Starting CelebA Pipeline... Press 'q' to quit.")
    
    frame_id = 0
    prev_time = 0
    score_buffer = deque(maxlen=30)
    smoother = ProbabilitySmoother(alpha=0.3)
    cooldown = AlertCooldown(cooldown_time=1.5)
    micro_alert_until = 0
    micro_label = ""
    
     # Session Statistics
    # Pre-populate to show all keys even if 0
    emotion_counts = defaultdict(int) 
    all_keys = [k.title() for k in EMOTION_MAP.keys()] + ["Neutral", "Unknown"]
    for k in all_keys:
        emotion_counts[k] = 0
        
    total_processed_frames = 0
    start_time = time.time()
    
    try:
        while True:
            if args.duration > 0 and (time.time() - start_time) > args.duration:
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
            
            results = yolo_model(frame, verbose=False, conf=CONF_THRESHOLD)
            best_face = None
            max_area = 0
            for result in results:
                for box in result.boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    h_img, w_img, _ = frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    if x1 >= x2 or y1 >= y2: continue

                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_face = (x1, y1, x2, y2)

            if best_face is not None:
                should_draw = isinstance(source, int) or args.view
                micro_text, conf, current_emotion_name = process_face(
                    frame, best_face, resnet, fer, lstm_model, device, 
                    frame_id, current_time, csv_writer, score_buffer,
                    smoother, cooldown, draw=should_draw
                )
                
                if micro_text:
                    micro_label = micro_text
                    micro_alert_until = current_time + 1.5
                    
                # [STATS] Accumulate emotion for summary
                if micro_text:
                     clean_key = micro_text.replace("LSTM: ", "").title()
                     emotion_counts[clean_key] += 1
                else: 
                     clean_key = current_emotion_name.title() if current_emotion_name else "Neutral"
                     emotion_counts[clean_key] += 1
                
                total_processed_frames += 1

            if isinstance(source, int) or args.view:
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('CelebA Experiment Pipeline', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            else:
                if frame_id % 30 == 0:
                    print(f"Processing Frame {frame_id}...", end='\r')
                    
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file: csv_file.close()
        
        # Print Summary
        if total_processed_frames > 0:
            print("\n" + "="*40)
            print(f" SESSION SUMMARY ({time.time() - start_time:.1f}s)")
            print("="*40)
            print(f"Total Frames Analyzed: {total_processed_frames}")
            # Filter out keys with 0 if you want, OR keep them to prove they exist.
            # User wants to know "other segments exists", so we show all.
            sorted_keys = sorted(emotion_counts.keys())
            for emo in sorted_keys:
                count = emotion_counts[emo]
                pct = (count / total_processed_frames) * 100
                print(f"{emo:<20}: {pct:.1f}%")
            print("="*40 + "\n")
            
        print("\nPipeline stopped.")

if __name__ == "__main__":
    start_time = time.time()
    main()
