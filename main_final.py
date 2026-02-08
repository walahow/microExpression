"""
Hybrid Macro-Micro Expression Detection System
Combines HSEmotion (Macro) + LSTM Specialist (Micro) for comprehensive real-time analysis
"""

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

import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import sys
import os

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HSEMOTION_MODEL = os.path.join(PROJECT_ROOT, "enet_b0_8_best_vgaf.pt")
LSTM_MODEL = os.path.join(PROJECT_ROOT, "lstm_diff_specialist.pth")
YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "yolov8n-face.pt")

# Constants
SEQ_LEN = 15
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MACRO_CONFIDENCE_THRESHOLD = 0.75
MICRO_CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence to trigger micro-expression alert

# HSEmotion Class Mapping (8 classes)
HSEMOTION_CLASSES = {
    0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear',
    4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'
}

# LSTM Micro Class Mapping (3 classes)
MICRO_CLASSES = {
    0: 'Others',
    1: 'Disgust',
    2: 'Repression'
}


# --- Stabilization Helpers ---
class PredictionSmoother:
    """Temporal smoothing to prevent flickering predictions."""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.prediction_history = deque(maxlen=window_size)
    
    def update(self, prediction):
        """Add prediction and return most common recent prediction."""
        self.prediction_history.append(prediction)
        
        if len(self.prediction_history) < 3:
            return prediction  # Not enough history, return current
        
        # Return most common prediction in recent history
        from collections import Counter
        counter = Counter(self.prediction_history)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def clear(self):
        self.prediction_history.clear()


class DisplayStabilizer:
    """Prevents rapid flickering by requiring state changes to persist."""
    
    def __init__(self, hold_frames=10):
        self.hold_frames = hold_frames
        self.current_status = "INITIALIZING..."
        self.current_color = (200, 200, 200)
        self.frames_remaining = 0
    
    def update(self, new_status, new_color):
        """Update display, but only if hold time expired or same state."""
        if self.frames_remaining > 0:
            # Still holding previous state
            self.frames_remaining -= 1
            return self.current_status, self.current_color
        
        # Hold time expired, can update
        if new_status != self.current_status:
            # State changed, start new hold period
            self.current_status = new_status
            self.current_color = new_color
            self.frames_remaining = self.hold_frames
        
        return self.current_status, self.current_color


# --- Model Definitions ---
class LSTMDiffSpecialist(nn.Module):
    """Unidirectional LSTM for micro-expression classification using difference features."""
    
    def __init__(self, input_size=1280, hidden_size=128, num_layers=1, num_classes=3):
        super(LSTMDiffSpecialist, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(last_out)  # (batch, num_classes)
        return out


# --- Model Loading ---
def load_hsemotion(model_path, device):
    """Load HSEmotion model for both macro classification and feature extraction."""
    print(f"Loading HSEmotion from {model_path}...")
    
    if device == torch.device('cpu'):
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    else:
        model = torch.load(model_path, weights_only=False)
    
    # Save classifier weights BEFORE removing for macro predictions
    if isinstance(model.classifier, torch.nn.Sequential):
        classifier_weights = model.classifier[0].weight.cpu().data.numpy().copy()
        classifier_bias = model.classifier[0].bias.cpu().data.numpy().copy()
    else:
        classifier_weights = model.classifier.weight.cpu().data.numpy().copy()
        classifier_bias = model.classifier.bias.cpu().data.numpy().copy()
    
    # Replace classifier with Identity to extract 1280-dim features
    model.classifier = torch.nn.Identity()
    
    model = model.to(device)
    model.eval()
    
    print("HSEmotion loaded successfully (Classifier replaced with Identity)")
    return model, classifier_weights, classifier_bias


def load_lstm_specialist(model_path, device):
    """Load LSTM micro-expression specialist."""
    print(f"Loading LSTM specialist from {model_path}...")
    
    model = LSTMDiffSpecialist(
        input_size=1280,
        hidden_size=128,
        num_layers=1,
        num_classes=3
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("LSTM specialist loaded successfully")
    else:
        print(f"WARNING: LSTM weights not found at {model_path}")
    
    model.eval()
    return model


# --- Helper Functions ---
def get_macro_prediction(features, weights, bias):
    """Get macro expression prediction from HSEmotion features."""
    logits = np.dot(features, np.transpose(weights)) + bias
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    return pred_idx, confidence


def compute_difference_features(features):
    """
    Compute difference features: D_i = F_i - F_0
    
    Args:
        features: numpy array of shape (seq_len, feature_dim)
    
    Returns:
        diff_features: numpy array of shape (seq_len, feature_dim)
    """
    F_0 = features[0]  # Reference frame (first frame in buffer)
    diff_features = features - F_0  # Broadcasting
    return diff_features


# --- Main Application ---
def main():
    print("="*70)
    print("Hybrid Macro-Micro Expression Detection System")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    # 1. Load Models
    print("\n[1/4] Loading Models...")
    yolo = YOLO(YOLO_WEIGHTS)
    hsemotion, macro_weights, macro_bias = load_hsemotion(HSEMOTION_MODEL, DEVICE)
    lstm_specialist = load_lstm_specialist(LSTM_MODEL, DEVICE)
    
    # 2. Setup Camera
    print("\n[2/4] Initializing Camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {width}x{height}")
    
    # 3. Setup Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Initialize Buffers & Stabilization
    print("\n[3/4] Initializing Buffers...")
    face_buffer = deque(maxlen=SEQ_LEN)  # Stores cropped face images (RGB)
    feature_buffer = deque(maxlen=SEQ_LEN)  # Stores 1280-dim features
    
    # Stabilization
    prediction_smoother = PredictionSmoother(window_size=5)
    display_stabilizer = DisplayStabilizer(hold_frames=12)  # ~0.4 seconds at 30 FPS
    
    # State
    current_status = "INITIALIZING..."
    current_color = (200, 200, 200)  # Grey
    fps_counter = deque(maxlen=30)
    
    print("\n[4/4] Starting Real-Time Detection...")
    print("="*70)
    print("Controls:")
    print("  'q' - Quit")
    print("  'r' - Reset buffer")
    print("="*70)
    
    try:
        print("\nStarting main loop...")
        frame_count = 0
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame (frame_count={frame_count})")
                break
            
            frame_count += 1
            if frame_count == 1:
                print(f"First frame captured successfully: {frame.shape}")
            
            # Mirror for user friendliness
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # ================== STEP A: Face Detection ==================
            face_crop = None
            face_coords = None
            
            try:
                results = yolo(frame, verbose=False)
                
                if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    # Get largest face
                    boxes = results[0].boxes
                    areas = []
                    
                    for box_obj in boxes:
                        coords = box_obj.xyxy[0]
                        w = coords[2] - coords[0]
                        h = coords[3] - coords[1]
                        area = (w * h).item()
                        areas.append(area)
                    
                    max_idx = np.argmax(areas)
                    best_box = boxes[max_idx]
                    coords = best_box.xyxy[0]
                    
                    if coords.is_cuda:
                        coords = coords.cpu()
                    
                    box = coords.numpy().astype(int)
                    x1, y1, x2, y2 = box
                    
                    # Safety bounds
                    h_img, w_img = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    if x2 > x1 and y2 > y1:
                        face = frame[y1:y2, x1:x2]
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_crop = face_rgb
                        face_coords = (x1, y1, x2, y2)
            
            except Exception as e:
                print(f"Face detection error: {e}")
            
            # ================== Buffer Management ==================
            if face_crop is not None:
                face_buffer.append(face_crop)
                
                # Extract features from current face
                try:
                    img_tensor = transform(Image.fromarray(face_crop))
                    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        features = hsemotion(img_tensor)  # (1, 1280)
                        features_np = features.cpu().numpy()[0]  # (1280,)
                        feature_buffer.append(features_np)
                
                except Exception as e:
                    print(f"Feature extraction error: {e}")
            else:
                # No face detected - clear buffers
                face_buffer.clear()
                feature_buffer.clear()
                current_status = "NO FACE DETECTED"
                current_color = (100, 100, 100)  # Dark grey
            
            # ================== STEP B & C: Macro and Micro Inference ==================
            if len(feature_buffer) == SEQ_LEN:
                try:
                    # Get current frame features for macro prediction
                    current_features = feature_buffer[-1]  # Most recent frame
                    
                    # STEP B: Macro Inference (HSEmotion)
                    macro_pred_idx, macro_confidence = get_macro_prediction(
                        current_features, macro_weights, macro_bias
                    )
                    macro_label = HSEMOTION_CLASSES[macro_pred_idx]
                    
                    # STEP C: Micro Inference (LSTM with difference features)
                    features_array = np.array(list(feature_buffer))  # (15, 1280)
                    diff_features = compute_difference_features(features_array)
                    
                    # LSTM inference with confidence
                    input_tensor = torch.FloatTensor(diff_features).unsqueeze(0).to(DEVICE)  # (1, 15, 1280)
                    with torch.no_grad():
                        micro_logits = lstm_specialist(input_tensor)
                        micro_probs = torch.softmax(micro_logits, dim=1)  # Get probabilities
                        micro_confidence, micro_pred_idx = torch.max(micro_probs, dim=1)
                        micro_pred_idx = micro_pred_idx.item()
                        micro_confidence = micro_confidence.item()
                    
                    micro_label = MICRO_CLASSES[micro_pred_idx]
                    
                    # Apply smoothing to micro predictions
                    micro_label_smoothed = prediction_smoother.update(micro_label)
                    
                    # ================== HYBRID FUSION LOGIC ==================
                    """
                    Priority 1 (Macro): If confidence > 0.75 and NOT Neutral
                    Priority 2 (Micro): If Macro is Neutral or low confidence
                    Priority 3: Default to Neutral
                    """
                    
                    # Determine raw status (before stabilization)
                    raw_status = ""
                    raw_color = (200, 200, 200)
                    
                    if macro_confidence > MACRO_CONFIDENCE_THRESHOLD and macro_label != 'Neutral':
                        # High-confidence macro expression
                        raw_status = f"MACRO: {macro_label.upper()}"
                        raw_color = (0, 255, 0)  # Green
                    
                    elif macro_label == 'Neutral' or macro_confidence <= MACRO_CONFIDENCE_THRESHOLD:
                        # Macro is neutral or uncertain -> check micro (use smoothed label)
                        # BUT only if micro confidence exceeds threshold
                        if micro_confidence >= MICRO_CONFIDENCE_THRESHOLD:
                            if micro_label_smoothed == 'Disgust':
                                raw_status = "MICRO-DISGUST (Hidden)"
                                raw_color = (0, 0, 255)  # Red
                            
                            elif micro_label_smoothed == 'Repression':
                                raw_status = "REPRESSION (Suppressed)"
                                raw_color = (0, 165, 255)  # Orange
                            
                            elif micro_label_smoothed == 'Others':
                                raw_status = "MICRO-MOVEMENT"
                                raw_color = (0, 255, 255)  # Yellow
                            
                            else:
                                raw_status = "NEUTRAL"
                                raw_color = (200, 200, 200)  # Grey
                        else:
                            # Micro confidence too low -> default to neutral
                            raw_status = "NEUTRAL"
                            raw_color = (200, 200, 200)  # Grey
                    
                    else:
                        raw_status = "NEUTRAL"
                        raw_color = (200, 200, 200)  # Grey
                    
                    # Apply display stabilization (prevents flickering)
                    current_status, current_color = display_stabilizer.update(raw_status, raw_color)
                    
                    # Debug info (optional)
                    debug_info = f"Macro: {macro_label} ({macro_confidence:.2f}) | Micro: {micro_label_smoothed} ({micro_confidence:.2f})"
                    cv2.putText(display_frame, debug_info, (10, height - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                except Exception as e:
                    print(f"Inference error: {e}")
                    current_status = "ERROR"
                    current_color = (255, 0, 255)  # Magenta
            
            elif len(feature_buffer) > 0:
                current_status = f"INITIALIZING... ({len(feature_buffer)}/{SEQ_LEN})"
                current_color = (255, 165, 0)  # Orange
            
            # ================== UI Rendering ==================
            # Draw face bounding box
            if face_coords is not None:
                x1, y1, x2, y2 = face_coords
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), current_color, 2)
            
            # Main status (larger, prominent)
            cv2.putText(display_frame, current_status, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, current_color, 3)
            
            # Buffer status bar
            buffer_ratio = len(feature_buffer) / SEQ_LEN
            bar_x, bar_y, bar_w, bar_h = 10, 70, 200, 15
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                         (50, 50, 50), -1)
            cv2.rectangle(display_frame, (bar_x, bar_y),
                         (bar_x + int(bar_w * buffer_ratio), bar_y + bar_h),
                         current_color, -1)
            cv2.putText(display_frame, f"Buffer: {len(feature_buffer)}/{SEQ_LEN}",
                       (bar_x, bar_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # FPS
            frame_time = time.time() - frame_start
            fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
            avg_fps = np.mean(fps_counter)
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (width - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Hybrid Macro-Micro Expression Detector", display_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                face_buffer.clear()
                feature_buffer.clear()
                prediction_smoother.clear()
                print("Buffers reset")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        import traceback
        print("\nFATAL ERROR:")
        traceback.print_exc()
    
    finally:
        print("\nShutting down...")
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
