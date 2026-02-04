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
from lstm_model import MicroExpressionLSTM
from emotion_recognizer import HSEmotionRecognizer
import torchvision.models as models
from torchvision import transforms

# --- Configuration ---
DATA_ROOT = "data/CASME2/CASME2 Preprocessed v2" 
FEATURES_DIR = "data/features_celeba" # NEW directory for CelebA features
SEQUENCE_LENGTH = 30
INPUT_SIZE = 512 + 8  # 512 (CelebA ResNet) + 8 (Emotion Scores)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
MODEL_SAVE_PATH = "lstm_celeba.pth"
BACKBONE_PATH = "celeba_backbone.pth"

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
    
    # --- Split Train/Test (Subject Independent by ID Sorting) ---
    # We assume 'reg_imgXX' ids correspond roughly to subject timeline.
    # Sorting by ID prevents mixing frames/clips of the same subject between train/test.
    def get_file_id(fpath):
        fname = os.path.basename(fpath)
        match = re.search(r'reg_img(\d+)', fname)
        return int(match.group(1)) if match else 0
        
    feature_files.sort(key=get_file_id)
    
    split_idx = int(len(feature_files) * 0.8)
    train_files = feature_files[:split_idx]
    test_files = feature_files[split_idx:]
    
    if len(test_files) > 0:
        print(f"Train Range: {get_file_id(train_files[0])} - {get_file_id(train_files[-1])}")
        print(f"Test Range:  {get_file_id(test_files[0])} - {get_file_id(test_files[-1])}")
    # ------------------------------------------------------------
    
    train_dataset = CasmeDataset(train_files, SEQUENCE_LENGTH)
    test_dataset = CasmeDataset(test_files, SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MicroExpressionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
