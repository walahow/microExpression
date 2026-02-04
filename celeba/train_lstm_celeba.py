import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
import glob
import cv2
from tqdm import tqdm
import sys
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lstm_model import MicroExpressionLSTM
from emotion_recognizer import HSEmotionRecognizer
import torchvision.models as models
from torchvision import transforms

# --- Configuration ---
# Get absolute path to the project root (one level up from this script)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "CASME2", "CASME2 Preprocessed v2")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features_celeba")
# Sequence & Model Params
SEQUENCE_LENGTH = 30
INPUT_SIZE = 512 + 8 
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Checkpoints (Save to same dir as script)
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "lstm_celeba.pth")
BACKBONE_PATH = os.path.join(os.path.dirname(__file__), "celeba_backbone.pth")

# Map CASME II folders to Labels
EMOTION_MAP = {
    'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 
    'fear': 4, 'sadness': 5, 'others': 6
}
NUM_CLASSES = len(EMOTION_MAP)

# --- Preprocessing Step ---
def extract_features_from_dataset():
    """
    Scans DATA_ROOT, extracts emotion scores AND CelebA-ResNet embeddings for every frame, 
    and saves them as .npy files in FEATURES_DIR.
    """
    if not os.path.exists(DATA_ROOT):
        print(f"Dataset not found at {DATA_ROOT}. Please download CASME II.")
        return False

    if not os.path.exists(FEATURES_DIR):
        print(f"Creating features directory: {FEATURES_DIR}")
        os.makedirs(FEATURES_DIR)
        
    print("Initializing Feature Extractor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Emotion Config
    try:
        fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
        print("HSEmotionRecognizer initialized successfully.")
    except Exception as e:
        print(f"WARNING: HSEmotionRecognizer failed to load: {e}")
        fer = None
        
    # 2. CelebA Config (ResNet18)
    print(f"Loading CelebA Backbone from {BACKBONE_PATH}...")
    try:
        # Recreate the structure used in training
        resnet = models.resnet18(pretrained=False)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 40) # Match the saved checkpoint structure
        
        # Load weights
        resnet.load_state_dict(torch.load(BACKBONE_PATH))
        
        # Remove the classification head (fc) to get 512 features
        # ResNet18 structure: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # We want everything up to avgpool.
        modules = list(resnet.children())[:-1] 
        resnet = nn.Sequential(*modules)
        
        resnet.eval().to(device)
        print("CelebA ResNet18 backbone loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load CelebA backbone: {e}")
        return False
    
    # Normalization for ResNet
    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    print(f"Scanning {DATA_ROOT}...")
    # Walk through emotion folders
    for emotion_name, label_idx in EMOTION_MAP.items():
        emotion_dir = os.path.join(DATA_ROOT, emotion_name)
        if not os.path.isdir(emotion_dir):
            continue
            
        print(f"Processing {emotion_name}...")
        
        # 1. Collect all images
        image_files = glob.glob(os.path.join(emotion_dir, "*.jpg"))
        
        # 2. Group by Video ID
        video_sequences = {} # { "base_name": [ (order_idx, filepath), ... ] }
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]
            
            if "(" in name_no_ext and ")" in name_no_ext:
                base_part, idx_part = name_no_ext.rsplit('(', 1)
                base_name = base_part.strip()
                idx_str = idx_part.replace(')', '').strip()
                try:
                    idx = int(idx_str)
                except:
                    idx = 0 
            else:
                base_name = name_no_ext.strip()
                idx = 1
                
            if base_name not in video_sequences:
                video_sequences[base_name] = []
            video_sequences[base_name].append((idx, img_path))
            
        print(f"  Found {len(video_sequences)} unique sequences in {emotion_name}.")

        # 3. Process Each Sequence
        for base_name, frames_list in tqdm(video_sequences.items(), desc=f"Extracting {emotion_name}"):
            # Sort by index
            frames_list.sort(key=lambda x: x[0])
            sorted_paths = [x[1] for x in frames_list]
            
            # Skip if sequence is too short
            if len(sorted_paths) < 3:
                continue

            # Save format: {emotion_name}_{base_name}.npy
            save_path = os.path.join(FEATURES_DIR, f"{emotion_name}_{base_name}.npy")
            if os.path.exists(save_path):
                continue 

            frame_imgs = []
            face_tensors = []
            
            # Prepare Data
            for frame_path in sorted_paths:
                img = cv2.imread(frame_path)
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_imgs.append(img_rgb)
                
                # Resize for ResNet (224x224)
                face_img = cv2.resize(img_rgb, (224, 224))
                ft = torch.from_numpy(face_img).float().permute(2, 0, 1) / 255.0
                ft = norm_transform(ft)
                face_tensors.append(ft)
            
            if not face_tensors:
                continue
                
            # Batch Inference: CelebA ResNet
            # Stack all frames into one batch [SeqLen, 3, 224, 224]
            # If SeqLen is very large, might OOM, but CASMEII is usually small (<100 frames).
            batch_tensor = torch.stack(face_tensors).to(device)
            
            embeddings = []
            with torch.no_grad():
                # Process in chunks of 32 to be safe
                for i in range(0, len(batch_tensor), 32):
                    batch = batch_tensor[i:i+32]
                    emb = resnet(batch).view(batch.size(0), -1).cpu().numpy()
                    embeddings.extend(emb)
            
            embeddings = np.array(embeddings)
            
            # Inference: Emotion (Sequential as FER might not handle batching seamlessly)
            scores_list = []
            if fer is not None:
                for img_rgb in frame_imgs:
                     _, s = fer.predict_emotions(img_rgb, logits=False)
                     scores_list.append(s)
            else:
                scores_list = [np.zeros(8) for _ in frame_imgs]
            
            # Combine
            sequence_features = []
            min_len = min(len(embeddings), len(scores_list))
            for i in range(min_len):
                combined = np.concatenate((embeddings[i], scores_list[i]))
                sequence_features.append(combined)
            
            if len(sequence_features) > 0:
                np.save(save_path, np.array(sequence_features))
                
    print(f"Feature extraction complete. Saved to {FEATURES_DIR}")
    return True

# --- Dataset Loader ---
class CasmeDataset(Dataset):
    def __init__(self, file_list, sequence_length=30):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        
        for file_path in file_list:
            # Parse label from filename: {emotion}_{basename}.npy
            filename = os.path.basename(file_path)
            emotion = filename.split('_')[0]
            if emotion not in EMOTION_MAP:
                continue
            label = EMOTION_MAP[emotion]
            
            # Load features
            features = np.load(file_path)
            
            # Pad or Truncate to SEQUENCE_LENGTH
            if len(features) >= sequence_length:
                # Truncate (take middle or first?) -> Standard is usually first or uniform sample
                features = features[:sequence_length]
            else:
                # Pad with zeros
                padding = np.zeros((sequence_length - len(features), INPUT_SIZE))
                features = np.concatenate((features, padding))
            
            self.data.append(features)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# --- Training Loop ---
def train_model():
    print("--- Starting LSTM Training (CelebA Backbone Experiment) ---")
    
    # 1. Ensure Features Exist
    if not extract_features_from_dataset():
        print("Feature extraction failed or dataset missing.")
        return

    # 2. Prepare Data
    feature_files = glob.glob(os.path.join(FEATURES_DIR, "*.npy"))
    if len(feature_files) == 0:
        print("No features found. Check dataset path.")
        return
        
    print(f"Found {len(feature_files)} sequences.")
    
    print(f"Found {len(feature_files)} sequences.")
    
    # --- Split Train/Test (Subject Independent using Map) ---
    import random
    import json
    
    # Load Subject Map (Recovered from Face Clustering)
    map_path = os.path.join(PROJECT_ROOT, "subject_map.json")
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            raw_map = json.load(f)
            SUBJECT_MAP = {int(k): v for k, v in raw_map.items()}
        print(f"Loaded Subject Map with {len(SUBJECT_MAP)} entries.")
    else:
        print("WARNING: subject_map.json not found. Falling back to filename ID (Leaky!).")
        SUBJECT_MAP = {}

    def get_subject_id(fpath):
         fname = os.path.basename(fpath)
         # Try matching reg_imgXX first (CelebA extracted might keep original names or not?)
         # The files are named {emotion}_{basename}.npy 
         # basename usually contains reg_imgXX
         match = re.search(r'(\d+)', fname)
         if match:
             vid_id = int(match.group(1))
             return SUBJECT_MAP.get(vid_id, vid_id) 
         return 0

    # 1. Get all unique subjects
    all_subjects = set(get_subject_id(f) for f in feature_files)
    all_subjects = list(all_subjects)
    all_subjects.sort() 
    
    # 2. Shuffle Subjects
    random.seed(42)
    random.shuffle(all_subjects)
    
    # 3. Split Subjects
    split_idx = int(0.8 * len(all_subjects))
    train_subject_ids = set(all_subjects[:split_idx])
    test_subject_ids = set(all_subjects[split_idx:])
    
    # 4. Assign Files
    train_files = [f for f in feature_files if get_subject_id(f) in train_subject_ids]
    test_files = [f for f in feature_files if get_subject_id(f) in test_subject_ids]
    
    print(f"Total Detected Subjects: {len(all_subjects)}")
    print(f"Train Subjects: {len(train_subject_ids)} | Test Subjects: {len(test_subject_ids)}")
    print(f"Train Files: {len(train_files)} | Test Files: {len(test_files)}")
    
    train_dataset = CasmeDataset(train_files, SEQUENCE_LENGTH)
    test_dataset = CasmeDataset(test_files, SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MicroExpressionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Add Weight Decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 4. Train
    best_acc = 0.0
    
    print(f"Training on device: {device}")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs, _ = model(features)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print(f"Training Complete. Best Test Acc: {best_acc:.2f}%")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    with open("result.txt", "w") as f:
        f.write(f"{best_acc:.2f}")

if __name__ == "__main__":
    train_model()
