import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import json
import re
from sklearn.model_selection import KFold
import sys
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lstm_model import MicroExpressionLSTM
from train_lstm import CasmeDataset, EMOTION_MAP, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, SEQUENCE_LENGTH, FEATURES_DIR as ORG_FEATURES_DIR

# Override paths to be absolute
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features_v2")

# Configuration
K_FOLDS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
# We can use fewer epochs for CV if it converges fast, but let's stick to 40-50 to be sure.
NUM_EPOCHS = 40 

def get_subject_id(fpath, subject_map):
    fname = os.path.basename(fpath)
    match = re.search(r'(\d+)', fname)
    if match:
        vid_id = int(match.group(1))
        return subject_map.get(vid_id, vid_id)
    return 0

def run_cv():
    print(f"--- Starting {K_FOLDS}-Fold Subject-Independent Cross Validation ---")
    
    # 1. Load Data Availability
    all_files = glob.glob(os.path.join(FEATURES_DIR, "*.npy"))
    if len(all_files) == 0:
        print("No features found. Run train_lstm.py first to extract.")
        return

    # 2. Load Subject Map
    map_path = os.path.join(PROJECT_ROOT, "subject_map.json")
    subject_map = {}
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            raw_map = json.load(f)
            subject_map = {int(k): v for k, v in raw_map.items()}
    
    # 3. Group Files by Subject
    subject_to_files = {}
    for f in all_files:
        sid = get_subject_id(f, subject_map)
        if sid not in subject_to_files:
            subject_to_files[sid] = []
        subject_to_files[sid].append(f)
        
    unique_subjects = list(subject_to_files.keys())
    unique_subjects.sort()
    
    print(f"Found {len(unique_subjects)} Subjects: {unique_subjects}")
    
    # 4. K-Fold Split
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_subjects)):
        print(f"\n=== Fold {fold+1}/{K_FOLDS} ===")
        
        train_subs = [unique_subjects[i] for i in train_idx]
        test_subs = [unique_subjects[i] for i in test_idx]
        
        # Gather files
        train_files = []
        for s in train_subs:
            train_files.extend(subject_to_files[s])
            
        test_files = []
        for s in test_subs:
            test_files.extend(subject_to_files[s])
            
        print(f"Train Subjects: {len(train_subs)}, Test Subjects: {len(test_subs)}")
        print(f"Train Files: {len(train_files)}, Test Files: {len(test_files)}")
        
        # Dataset & Loader
        train_dataset = CasmeDataset(train_files, SEQUENCE_LENGTH)
        test_dataset = CasmeDataset(test_files, SEQUENCE_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model
        model = MicroExpressionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Same optimized config
        
        best_fold_acc = 0.0
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Val
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, _ = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
        
        print(f"Fold {fold+1} Best Accuracy: {best_fold_acc:.2f}%")
        fold_results.append(best_fold_acc)
        
    # 5. Final Report
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    print("\n\n=== Cross Validation Results ===")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1}: {acc:.2f}%")
    print(f"---------------------------")
    print(f"Mean Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
    
if __name__ == "__main__":
    run_cv()
