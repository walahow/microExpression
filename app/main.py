import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import sys
import os

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FFHQ = os.path.join(PROJECT_ROOT, "..", "celeba", "weights_ffhq_pretrained.pth")
WEIGHTS_LSTM = os.path.join(PROJECT_ROOT, "lstm_v3_apex_balanced.pth")
YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "yolov8n-face.pt")

# Constants
SEQ_LEN = 15
CROP_SIZE = 224
MOTION_THRESHOLD = 2.0 # Adjustable via keys
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions ---
# 1. ResNet Backbone
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet18(weights=None) # Start blank
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: [B, 512, 1, 1]
        
    def forward(self, x):
        return self.backbone(x).flatten(start_dim=1)

# 2. LSTM Model
class UniDirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(UniDirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.fc(out)
        return out

# --- Helper Functions ---
def load_resnet(path):
    model = ResNetBackbone().to(DEVICE)
    if os.path.exists(path):
        print(f"Loading FFHQ weights from {path}...")
        try:
            state_dict = torch.load(path, map_location=DEVICE)
            # Fix keys if needed (remove 'encoder.' prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    new_state_dict[k.replace('encoder.', '')] = v
                else:
                    new_state_dict[k] = v
            model.backbone.load_state_dict(new_state_dict, strict=False)
            print("ResNet weights loaded.")
        except Exception as e:
            print(f"Error loading ResNet: {e}")
    else:
        print(f"Warning: ResNet weights not found at {path}")
    model.eval()
    return model

def load_lstm(path):
    # 7 Classes: Happy, Surprise, Disgust, Repression, Fear, Sadness, Others
    model = UniDirectionalLSTM(input_size=512, hidden_size=128, num_layers=1, num_classes=7).to(DEVICE)
    if os.path.exists(path):
        print(f"Loading LSTM weights from {path}...")
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            print("LSTM weights loaded.")
        except Exception as e:
            print(f"Error loading LSTM: {e}")
    else:
        print(f"Critical: LSTM weights not found at {path}")
    model.eval()
    return model

def compute_optical_flow(prev, curr):
    if prev is None or curr is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate Magnitude for Gating
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = np.mean(mag)
    
    # Visualization (Features)
    flow_norm = np.zeros_like(prev)
    flow_norm[..., 1] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    flow_norm[..., 2] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    flow_norm[..., 0] = 0
    
    return flow_norm, mean_mag

# --- Main App ---
def main():
    print("Initializing App...")
    
    # 1. Load Models
    yolo = YOLO(YOLO_WEIGHTS)
    resnet = load_resnet(WEIGHTS_FFHQ)
    lstm = load_lstm(WEIGHTS_LSTM)
    
    # 2. Setup Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {width}x{height}")
    
    # 3. State & Buffers
    frame_buffer = deque(maxlen=SEQ_LEN) # Stores cropped face images
    motion_history = deque(maxlen=SEQ_LEN) # Stores motion scores
    flow_buffer = deque(maxlen=SEQ_LEN) # Stores computed optical flows
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Detection State
    last_status = "NEUTRAL"
    last_color = (200, 200, 200) # Grey
    display_timer = 0
    DISPLAY_HOLD_TIME = 20 # Frames to hold the alert
    
    global MOTION_THRESHOLD
    
    print("Starting loop. Press 'q' to quit.")
    print("Adjust Threshold: '+' to increase, '-' to decrease.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Mirror for user friendliness
            frame = cv2.flip(frame, 1)
            plot_frame = frame.copy()
            
            # A. Face Detection
            try:
                results = yolo(frame, verbose=False)
            except Exception as e:
                print(f"YOLO Inference Error: {e}")
                results = []

            face_crop = None
            face_coords = None
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                # Get largest face
                boxes = results[0].boxes
                
                # Safe Area Conv
                areas = []
                for box_obj in boxes:
                    # box_obj.xyxy is [1, 4] tensor
                    coords = box_obj.xyxy[0] # [4] tensor
                    w = coords[2] - coords[0]
                    h = coords[3] - coords[1]
                    area = (w * h).item()
                    areas.append(area)
                    
                max_idx = np.argmax(areas)
                
                # Safe Box Extraction
                # boxes[i] returns a Boxes object. 
                # .xyxy returns a tensor.
                # We need to be very explicit
                best_box = boxes[max_idx]
                coords = best_box.xyxy[0] # [4]
                if coords.is_cuda:
                    coords = coords.cpu()
                box = coords.numpy().astype(int)
                
                x1, y1, x2, y2 = box
                face_coords = (x1, y1, x2, y2)
                
                # Crop & Resize
                # Add margin?
                h_img, w_img = frame.shape[:2]
                # Safety check
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                if x2 > x1 and y2 > y1:
                    face = frame[y1:y2, x1:x2]
                    face_crop = cv2.resize(face, (CROP_SIZE, CROP_SIZE))
                    
                    # Draw Box
                    cv2.rectangle(plot_frame, (x1, y1), (x2, y2), last_color, 2)
            
            # B. Buffer Management
            current_motion = 0.0
            
            if face_crop is not None:
                frame_buffer.append(face_crop)
                
                if len(frame_buffer) > 1:
                    # Compute flow with previous frame
                    # Note: We compute flow relative to buffer[-2]
                    flow_img, mag = compute_optical_flow(frame_buffer[-2], frame_buffer[-1])
                    current_motion = mag
                    motion_history.append(mag)
                    flow_buffer.append(flow_img)
            else:
                # Clear buffer if face lost
                frame_buffer.clear()
                motion_history.clear()
                flow_buffer.clear()
                
            # C. Inference Logic
            avg_motion = np.mean(motion_history) if len(motion_history) > 0 else 0.0
            
            is_gated = avg_motion < MOTION_THRESHOLD
            
            # We need SEQ_LEN - 1 flows to theoretically match the time dimension, 
            # but our model expects SEQ_LEN inputs.
            # Original code padded the last one.
            if len(flow_buffer) == SEQ_LEN - 1 and not is_gated:
                # --- RUN INFERENCE ---
                
                # 1. Prepare Flows
                # Flows are already computed incrementally!
                flows = list(flow_buffer)
                # Pad to match SEQ_LEN (15)
                flows.append(flows[-1])
                
                # 2. Extract Features (ResNet)
                batch_tensors = []
                for f_img in flows:
                    tensor = transform(f_img)
                    batch_tensors.append(tensor)
                
                batch_input = torch.stack(batch_tensors).to(DEVICE)
                with torch.no_grad():
                    features = resnet(batch_input) # [15, 512]
                    features = features.unsqueeze(0) # [1, 15, 512]
                    
                    # 3. Predict (LSTM)
                    logits = lstm(features) # [1, 7]
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    conf = probs[0, pred_idx].item()
                    
                # D. Display Logic (Binary Mode)
                # Class 2 = Disgust
                # Others = 0,1,3,4,5,6
                
                if pred_idx == 2:
                    last_status = "DISGUST DETECTED"
                    last_color = (0, 0, 255) # Red
                else:
                    last_status = "MICRO-EXPRESSION DETECTED"
                    last_color = (0, 255, 255) # Yellow
                    
                display_timer = DISPLAY_HOLD_TIME
                
            # Display Timer Decay
            if display_timer > 0:
                display_timer -= 1
            else:
                last_status = "NEUTRAL"
                last_color = (200, 200, 200) # Grey
                
            # --- UI Overlay ---
            
            # 1. Motion Bar
            bar_x = 20
            bar_y = 60
            bar_w = 200
            bar_h = 20
            
            # Normalize motion for display (e.g. 0 to 10)
            motion_ratio = min(avg_motion / 10.0, 1.0)
            fill_w = int(bar_w * motion_ratio)
            
            bar_color = (0, 255, 0) if not is_gated else (0, 0, 255) # Green if active, Red if gated
            
            cv2.rectangle(plot_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(plot_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
            cv2.putText(plot_frame, f"Motion: {avg_motion:.2f} / {MOTION_THRESHOLD:.1f}", (bar_x, bar_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 2. Main Status
            cv2.putText(plot_frame, last_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)
            
            # 3. FPS
            cv2.imshow("Micro-Expression Detector (Binary Mode)", plot_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                MOTION_THRESHOLD += 0.5
            elif key == ord('-') or key == ord('_'):
                MOTION_THRESHOLD = max(0.5, MOTION_THRESHOLD - 0.5)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
