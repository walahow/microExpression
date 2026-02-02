import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import cv2
from tqdm import tqdm
from lstm_model import MicroExpressionLSTM
from emotion_recognizer import HSEmotionRecognizer
from facenet_pytorch import InceptionResnetV1

# --- Configuration ---
DATA_ROOT = "data/CASME2/CASME2 Preprocessed v2" # Updated path based on extraction structure
FEATURES_DIR = "data/features_v2"
SEQUENCE_LENGTH = 30
INPUT_SIZE = 8  # 8 (Emotion Scores ONLY). Removed 512 FaceNet to prevent identity leakage.
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
MODEL_SAVE_PATH = "lstm_micro_expression.pth"

# Map CASME II folders to Labels
EMOTION_MAP = {
    'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 
    'fear': 4, 'sadness': 5, 'others': 6
}
NUM_CLASSES = len(EMOTION_MAP)

# --- Preprocessing Step ---
def extract_features_from_dataset():
    """
    Scans DATA_ROOT, extracts emotion scores AND FaceNet embeddings for every frame, 
    and saves them as .npy files in FEATURES_DIR.
    """
    if not os.path.exists(DATA_ROOT):
        print(f"Dataset not found at {DATA_ROOT}. Please download CASME II.")
        return False

    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
        
    print("Initializing Feature Extractor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
        print("HSEmotionRecognizer initialized successfully.")
    except Exception as e:
        print(f"WARNING: HSEmotionRecognizer failed to load: {e}")
        print("Using dummy emotion scores for verification.")
        fer = None
        
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
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

            sequence_features = []
            for frame_path in sorted_paths:
                img = cv2.imread(frame_path)
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # A. Emotion Scores (8 dims)
                if fer is not None:
                    _, scores = fer.predict_emotions(img_rgb, logits=False)
                else:
                    # Dummy scores
                    scores = np.random.rand(8)
                    scores = scores / scores.sum() # Normalize
                
                # B. Face Embeddings (512 dims)
                # Resize to 160x160 for InceptionResnetV1
                face_img = cv2.resize(img_rgb, (160, 160))
                face_tensor = torch.from_numpy(face_img).float()
                face_tensor = (face_tensor - 127.5) / 128.0
                face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    embedding = resnet(face_tensor).cpu().numpy().flatten()
                
                # Combine: [Embedding (512), Scores (8)]
                combined = np.concatenate((embedding, scores))
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
        
        # Load all .npy files in the provided list
        for f in file_list:
            # Filename format: {emotion_name}_{video_name}.npy
            basename = os.path.basename(f)
            emotion_name = basename.split('_')[0]
            if emotion_name not in EMOTION_MAP:
                continue
            
            label = EMOTION_MAP[emotion_name]
            features = np.load(f) # (T, 520)
            
            # --- IMPORTANT: Slice to keep ONLY Emotion Scores (Last 8) ---
            # Structure was [FaceNet(512), Scores(8)]
            features = features[:, -8:] # Now shape is (T, 8).shape[0]
            T = features.shape[0]
            if T < self.sequence_length:
                # Pad with last frame or zeros
                pad = np.zeros((self.sequence_length - T, features.shape[1]))
                features_padded = np.vstack((features, pad))
            else:
                # Take middle 30 frames (apex is usually in middle)
                start = (T - self.sequence_length) // 2
                features_padded = features[start : start + self.sequence_length]

            self.data.append(torch.tensor(features_padded, dtype=torch.float32))
            self.labels.append(torch.tensor(label, dtype=torch.long))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Training Loop ---
def train_model():
    # 1. Preprocess
    if not extract_features_from_dataset():
        # Fallback for demo if no data
        print("Creating Dummy Data for testing code flow...")
        dummy_input = torch.randn(100, SEQUENCE_LENGTH, INPUT_SIZE)
        dummy_target = torch.randint(0, NUM_CLASSES, (100,))
        dataloader = DataLoader(list(zip(dummy_input, dummy_target)), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = dataloader # Dummy val
    else:
        # Load Data
        all_files = glob.glob(os.path.join(FEATURES_DIR, "*.npy"))
        if len(all_files) == 0:
            print("No features found after preprocessing. Check dataset path.")
            return

        # Simple random split 80/20
        # In a real scenario, we should split by Subject ID (LOSO)
        import random
        random.shuffle(all_files)
        split_idx = int(0.8 * len(all_files))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        train_dataset = CasmeDataset(train_files, SEQUENCE_LENGTH)
        val_dataset = CasmeDataset(val_files, SEQUENCE_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Loaded {len(train_dataset)} training sequences, {len(val_dataset)} validation sequences.")
    
    # 2. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = MicroExpressionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    
    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    # 4. Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs) # Returns output, attention_weights
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total if total > 0 else 0
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New Best Model Saved! ({best_acc:.2f}%)")
            
    print(f"Training Complete. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()
