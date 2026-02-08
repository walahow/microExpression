import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import glob

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "..", "data", "CASME2", "Cropped")
EXCEL_PATH = os.path.join(PROJECT_ROOT, "..", "data", "CASME2", "CASME2-coding-20140508.xlsx")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "..", "celeba", "weights_ffhq_pretrained.pth")
SAVE_FEATURES_PATH = os.path.join(PROJECT_ROOT, "casme_features_v3.npy")
SAVE_LABELS_PATH = os.path.join(PROJECT_ROOT, "casme_labels_v3.npy")
ERROR_LOG_PATH = os.path.join(PROJECT_ROOT, "extraction_errors.log")

# Parameters
SEQ_LEN = 15
STEP_SIZE = 6 # Simulate 30fps from ~200fps
CROP_SIZE = 224
ORIGINAL_H, ORIGINAL_W = 231, 282 # Approximate, will verify per image

# Emotion Label Map (Standard CASME II)
EMOTION_MAP = {
    'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 
    'fear': 4, 'sadness': 5, 'others': 6
}

# --- 1. Define Backbone (ResNet18) ---
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet18(pretrained=False)
        # Remove FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: [B, 512, 1, 1]
        
    def forward(self, x):
        return self.backbone(x).flatten(start_dim=1)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path):
    device = get_device()
    model = ResNetBackbone().to(device)
    
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    new_state_dict[k.replace('encoder.', '')] = v
                else:
                    new_state_dict[k] = v
            
            model.backbone.load_state_dict(new_state_dict, strict=False)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"WARNING: Failed to load weights: {e}")
    else:
        print(f"WARNING: Weights not found at {weights_path}. Using random init.")
    
    model.eval()
    return model

# --- 2. Image Processing ---
def center_crop(img, size=224):
    h, w, _ = img.shape
    cy, cx = h // 2, w // 2
    x1 = cx - size // 2
    y1 = cy - size // 2
    
    # Safety Check
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    x2 = x1 + size
    y2 = y1 + size
    
    if x2 > w: 
        x2 = w
        x1 = w - size
    if y2 > h: 
        y2 = h
        y1 = h - size
        
    return img[y1:y2, x1:x2]

def compute_optical_flow(prev, curr):
    # Convert to gray
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    
    # Farneback Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Visualize Flow (Normalize to 0-255 RGB)
    flow_norm = np.zeros_like(prev)
    flow_norm[..., 1] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX) # G = dx
    flow_norm[..., 2] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX) # R = dy
    flow_norm[..., 0] = 0 # B
    
    return flow_norm

# --- 3. Main Extraction Loop ---
def main():
    # Load Excel
    if not os.path.exists(EXCEL_PATH):
        excel_candidates = glob.glob(os.path.join(PROJECT_ROOT, "..", "data", "CASME2", "*.xlsx"))
        if excel_candidates:
            EXCEL_PATH_FINAL = excel_candidates[0]
            print(f"Found Excel at {EXCEL_PATH_FINAL}")
        else:
            print("CRITICAL: Excel file not found.")
            return
    else:
        EXCEL_PATH_FINAL = EXCEL_PATH
        
    try:
        df = pd.read_excel(EXCEL_PATH_FINAL)
    except ImportError:
         print("Error: `openpyxl` not installed. Please install it.")
         return

    # Check Columns
    df.columns = [c.strip() for c in df.columns]
    
    # Initialize Model
    device = get_device()
    model = load_model(WEIGHTS_PATH)
    
    transform = transforms.Compose([
        transforms.ToTensor(), # HWC->CHW, 0-1
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_features = []
    all_labels = []
    
    print(f"Processing {len(df)} videos...")
    
    skipped = 0
    errors = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            subject_id = int(row['Subject'])
            video_name = str(row['Filename'])
            
            # Robust parsing of ApexFrame
            try:
                apex_frame = int(row['ApexFrame'])
            except ValueError:
                skipped += 1
                continue

            emotion = str(row['Estimated Emotion'])
            if emotion not in EMOTION_MAP:
                continue
            
            label_idx = EMOTION_MAP[emotion]
            
            # Construct Path
            sub_folder = f"sub{subject_id:02d}"
            video_path = os.path.join(DATA_ROOT, sub_folder, video_name)
            
            if not os.path.exists(video_path):
                skipped += 1
                continue
                
            # Get Frame indices
            indices = []
            center_idx = 7
            for i in range(SEQ_LEN):
                offset = (i - center_idx) * STEP_SIZE
                frame_num = apex_frame + offset
                indices.append(frame_num)
                
            # Load Images
            images = []
            valid_paths = []
            for idx in indices:
                fname = f"reg_img{idx}.jpg"
                fpath = os.path.join(video_path, fname)
                valid_paths.append(fpath)
                
            loaded_imgs = []
            for fpath in valid_paths:
                img = cv2.imread(fpath)
                if img is not None:
                    img = center_crop(img, CROP_SIZE)
                    loaded_imgs.append(img)
                else:
                    loaded_imgs.append(None)
                    
            # Padding
            last_valid = None
            for i in range(len(loaded_imgs)):
                if loaded_imgs[i] is not None:
                    last_valid = loaded_imgs[i]
                elif last_valid is not None:
                    loaded_imgs[i] = last_valid.copy()
                    
            first_valid = None
            for i in range(len(loaded_imgs)-1, -1, -1):
                if loaded_imgs[i] is not None:
                    first_valid = loaded_imgs[i]
                elif first_valid is not None:
                    loaded_imgs[i] = first_valid.copy()
                    
            if any(img is None for img in loaded_imgs):
                skipped += 1
                continue
                
            # Compute Flow
            flows = []
            for i in range(len(loaded_imgs) - 1):
                prev = loaded_imgs[i]
                curr = loaded_imgs[i+1]
                flow = compute_optical_flow(prev, curr)
                flows.append(flow)
                
            if len(flows) < SEQ_LEN:
                flows.append(flows[-1])
                
            # Feature Extraction
            batch_tensors = []
            for flow_img in flows:
                flow_img = flow_img.astype(np.uint8)
                img_pil = transforms.ToPILImage()(flow_img)
                tensor = transform(img_pil)
                batch_tensors.append(tensor)
                
            batch_input = torch.stack(batch_tensors).to(device)
            
            with torch.no_grad():
                features = model(batch_input)
                
            features_np = features.cpu().numpy()
            
            all_features.append(features_np)
            all_labels.append(label_idx)
            
        except Exception as e:
            # Catch ALL errors for a single video to prevent script crash
            errors.append(f"Error processing video {row.get('Filename', 'UNKNOWN')}: {e}\n")
            skipped += 1
            continue
        
    # Stack and Save
    if len(all_features) > 0:
        X = np.array(all_features) 
        y = np.array(all_labels) 
        
        np.save(SAVE_FEATURES_PATH, X)
        np.save(SAVE_LABELS_PATH, y)
        print(f"Success! Processed {len(all_labels)} videos.")
        print(f"Skipped {skipped} videos.")
        
        if errors:
            with open(ERROR_LOG_PATH, 'w') as f:
                f.writelines(errors)
            print(f"See {ERROR_LOG_PATH} for details on {len(errors)} errors.")
    else:
        print("No features extracted. Check paths.")

if __name__ == "__main__":
    main()
