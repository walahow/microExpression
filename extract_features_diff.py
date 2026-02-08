import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "CASME2", "Cropped")
EXCEL_PATH = os.path.join(PROJECT_ROOT, "data", "CASME2", "CASME2-coding-20140508.xlsx")
MODEL_PATH = os.path.join(PROJECT_ROOT, "enet_b0_8_best_vgaf.pt")
SAVE_FEATURES_PATH = os.path.join(PROJECT_ROOT, "X_diff.npy")
SAVE_LABELS_PATH = os.path.join(PROJECT_ROOT, "y_diff.npy")

# Parameters
SEQ_LEN = 15
STEP_SIZE = 6  # Simulate 30fps from ~200fps CASME II
IMG_SIZE = 224

# Target Classes (Filter: Keep only these emotions)
TARGET_EMOTIONS = ['disgust', 'repression', 'others']

# Label Mapping
LABEL_MAP = {
    'others': 'Others',
    'disgust': 'Disgust', 
    'repression': 'Repression'
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hsemotion_backbone(model_path, device):
    """Load HSEmotion EfficientNet-B0 backbone and remove classifier."""
    print(f"Loading HSEmotion model from {model_path}...")
    
    if device == 'cpu':
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
    
    # Replace classifier with Identity to extract features
    model.classifier = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully. Classifier removed for feature extraction.")
    return model


def get_transform():
    """Get image preprocessing transform for EfficientNet-B0."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_available_frame_range(video_path):
    """
    Get the range of available frame indices in a video directory.
    
    Args:
        video_path: Path to video directory
    
    Returns:
        (min_frame, max_frame) or None if no frames found
    """
    if not os.path.exists(video_path):
        return None
    
    frame_files = [f for f in os.listdir(video_path) if f.startswith('reg_img') and f.endswith('.jpg')]
    if not frame_files:
        return None
    
    frame_indices = []
    for fname in frame_files:
        try:
            idx = int(fname.replace('reg_img', '').replace('.jpg', ''))
            frame_indices.append(idx)
        except ValueError:
            continue
    
    if not frame_indices:
        return None
    
    return min(frame_indices), max(frame_indices)


def extract_sequence_features_with_padding(model, transform, video_path, target_indices, device):
    """
    Extract features for a sequence of images with edge-repeating padding.
    
    If a target frame index is out of bounds, the nearest available frame is used.
    
    Args:
        model: HSEmotion backbone (without classifier)
        transform: Image preprocessing transform
        video_path: Path to video directory
        target_indices: List of target frame indices
        device: torch device
    
    Returns:
        features: numpy array of shape (seq_len, 1280) or None if failed
    """
    # Get available frame range
    frame_range = get_available_frame_range(video_path)
    if frame_range is None:
        return None
    
    min_frame, max_frame = frame_range
    
    features_list = []
    
    for target_idx in target_indices:
        # Clamp index to available range (edge-repeating padding)
        clamped_idx = max(min_frame, min(target_idx, max_frame))
        
        # Build image path
        fname = f"reg_img{clamped_idx}.jpg"
        img_path = os.path.join(video_path, fname)
        
        if not os.path.exists(img_path):
            return None  # Should not happen after clamping, but safety check
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform and add batch dimension
        img_tensor = transform(Image.fromarray(img_rgb))
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(img_tensor)
        
        features_np = features.cpu().numpy()[0]  # Remove batch dimension
        features_list.append(features_np)
    
    return np.array(features_list)  # Shape: (seq_len, 1280)


def compute_difference_features(features):
    """
    Compute difference features: D_i = F_i - F_0
    
    Args:
        features: numpy array of shape (seq_len, feature_dim)
    
    Returns:
        diff_features: numpy array of shape (seq_len, feature_dim)
    """
    F_0 = features[0]  # Reference frame (first frame in window)
    diff_features = features - F_0  # Broadcasting: subtract F_0 from each frame
    return diff_features


def main():
    device = get_device()
    print(f"Using device: {device}")
    
    # Load Excel metadata
    if not os.path.exists(EXCEL_PATH):
        print(f"ERROR: Excel file not found at {EXCEL_PATH}")
        return
    
    try:
        df = pd.read_excel(EXCEL_PATH)
    except ImportError:
        print("ERROR: openpyxl not installed. Please install it: pip install openpyxl")
        return
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded {len(df)} rows from Excel")
    
    # Filter for target emotions
    df['emotion_lower'] = df['Estimated Emotion'].str.lower().str.strip()
    df_filtered = df[df['emotion_lower'].isin(TARGET_EMOTIONS)].copy()
    print(f"Filtered to {len(df_filtered)} rows with target emotions: {TARGET_EMOTIONS}")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
    
    model = load_hsemotion_backbone(MODEL_PATH, device)
    transform = get_transform()
    
    # Extract features
    all_features = []
    all_labels = []
    skipped = 0
    
    print(f"\nExtracting features for {len(df_filtered)} videos...")
    
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        try:
            subject_id = int(row['Subject'])
            video_name = str(row['Filename']).strip()
            
            # Parse apex frame
            try:
                apex_frame = int(row['ApexFrame'])
            except (ValueError, TypeError):
                skipped += 1
                continue
            
            emotion = row['emotion_lower']
            label_str = LABEL_MAP[emotion]
            
            # Construct video path
            sub_folder = f"sub{subject_id:02d}"
            video_path = os.path.join(DATA_ROOT, sub_folder, video_name)
            
            if not os.path.exists(video_path):
                skipped += 1
                continue
            
            # Calculate frame indices (15 frames centered on apex)
            # Center index = 7 (0-indexed), so indices are:
            # [apex-42, apex-36, apex-30, ..., apex, ..., apex+30, apex+36, apex+42]
            center_idx = 7
            indices = []
            for i in range(SEQ_LEN):
                offset = (i - center_idx) * STEP_SIZE
                frame_num = apex_frame + offset
                indices.append(frame_num)
            
            # Extract features for sequence (with edge-repeating padding)
            features = extract_sequence_features_with_padding(model, transform, video_path, indices, device)
            
            if features is None:
                skipped += 1
                continue
            
            # Compute difference features
            diff_features = compute_difference_features(features)
            
            all_features.append(diff_features)
            all_labels.append(label_str)
            
        except Exception as e:
            print(f"\nError processing {row.get('Filename', 'UNKNOWN')}: {e}")
            skipped += 1
            continue
    
    # Save results
    if len(all_features) > 0:
        X = np.array(all_features)  # Shape: (N, 15, 1280)
        y = np.array(all_labels)    # Shape: (N,)
        
        np.save(SAVE_FEATURES_PATH, X)
        np.save(SAVE_LABELS_PATH, y)
        
        print(f"\n{'='*60}")
        print(f"SUCCESS! Extracted features from {len(all_labels)} videos")
        print(f"Skipped: {skipped} videos")
        print(f"\nOutput shapes:")
        print(f"  X_diff.npy: {X.shape}")
        print(f"  y_diff.npy: {y.shape}")
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count}")
        
        print(f"\nFiles saved to:")
        print(f"  {SAVE_FEATURES_PATH}")
        print(f"  {SAVE_LABELS_PATH}")
        print(f"{'='*60}")
    else:
        print("\nERROR: No features extracted. Check paths and data.")


if __name__ == "__main__":
    main()
