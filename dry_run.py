
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from lstm_model import MicroExpressionLSTM

# --- Configuration for Dry Run ---
FEATURES_DIR = "data/features_v2"
SEQUENCE_LENGTH = 30
INPUT_SIZE = 520 # Full features
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 1  
MODEL_SAVE_PATH = "lstm_dry_run.pth"

# Load Subject Map
map_path = "subject_map.json"
SUBJECT_MAP = {}
if os.path.exists(map_path):
    import json
    with open(map_path, 'r') as f:
        # Convert keys to int
        raw_map = json.load(f)
        SUBJECT_MAP = {int(k): v for k, v in raw_map.items()}

def get_subject_id(fpath):
    fname = os.path.basename(fpath)
    import re
    match = re.search(r'(\d+)', fname)
    if match:
        vid_id = int(match.group(1))
        return SUBJECT_MAP.get(vid_id, vid_id)
    return 0

EMOTION_MAP = {
    'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 
    'fear': 4, 'sadness': 5, 'others': 6
}
NUM_CLASSES = len(EMOTION_MAP)

class CasmeDataset(Dataset):
    def __init__(self, file_list, sequence_length=30):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        
        for f in file_list:
            basename = os.path.basename(f)
            emotion_name = basename.split('_')[0]
            if emotion_name not in EMOTION_MAP:
                continue
            
            label = EMOTION_MAP[emotion_name]
            features = np.load(f) 
            # features shape (T, 520) -> keep all
            features = features  
            
            T = features.shape[0]
            if T < self.sequence_length:
                pad = np.zeros((self.sequence_length - T, features.shape[1]))
                features_padded = np.vstack((features, pad))
            else:
                start = (T - self.sequence_length) // 2
                features_padded = features[start : start + self.sequence_length]

            self.data.append(torch.tensor(features_padded, dtype=torch.float32))
            self.labels.append(torch.tensor(label, dtype=torch.long))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_dry_run():
    all_files = glob.glob(os.path.join(FEATURES_DIR, "*.npy"))
    if len(all_files) == 0:
        print("No features found.")
        return

    # Use a small subset for speed
    subset_files = all_files[:20]
    
    train_dataset = CasmeDataset(subset_files, SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Dry Run: {len(train_dataset)} samples.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = MicroExpressionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Batch Loss: {loss.item():.4f}")
        
    print("Dry Run Complete. System is functional.")

if __name__ == "__main__":
    train_dry_run()
