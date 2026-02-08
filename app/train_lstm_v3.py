import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "casme_features_v3.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "casme_labels_v3.npy")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "lstm_v3_apex_balanced.pth")
CONFUSION_MATRIX_PATH = os.path.join(PROJECT_ROOT, "confusion_matrix_v3.png")

# Hyperparameters
INPUT_SIZE = 512
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Emotion Map (Standard CASME II - 7 Classes)
EMOTION_MAP = {
    0: 'Happiness', 1: 'Surprise', 2: 'Disgust', 3: 'Repression', 
    4: 'Fear', 5: 'Sadness', 6: 'Others'
}

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition ---
class UniDirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(UniDirectionalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=False)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_size]
        # LSTM output: [batch, seq_len, hidden_size]
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step for classification
        last_out = lstm_out[:, -1, :]
        
        out = self.dropout(last_out)
        out = self.fc(out)
        return out

def main():
    device = get_device()
    print(f"Training on device: {device}")
    
    # 1. Load Data
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(LABELS_PATH):
        print("Data not found!")
        return
        
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    
    print(f"Loaded Features: {X.shape}")
    print(f"Loaded Labels: {y.shape}")
    
    # 2. Compute Class Weights (1.0 / Count)
    unique_classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    print("Class Distribution:", class_counts)
    
    num_classes = len(unique_classes)
    
    # Ensure weights vector covers all possible classes (0 to max(y))
    # Standard CASME II has 7 classes (0-6)
    max_class = 6 
    weights = np.zeros(max_class + 1)
    
    for cls, count in class_counts.items():
        weights[cls] = 1.0 / count
        
    # Handle classes with 0 samples (though unlikely here)
    # If a class has 0 samples, weight is 0? Or effectively ignore.
    # But we found all classes present (though Fear only has 2).
    
    # Normalize weights? Not strictly necessary for CrossEntropyLoss but good practice
    weights = weights / weights.sum()
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Class Weights:", weights)
    
    # 3. Split Data
    # Stratify is important due to small classes (Fear=2)
    # But with only 2 samples, stratify might fail if test_size implies < 1 sample in test!
    # Try typical stratify first.
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("WARNING: Stratified split failed (too few samples in some class). Falling back to random split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
    print(f"Train Set: {X_train.shape[0]}")
    print(f"Val Set:   {X_val.shape[0]}")
    
    # Convert to Tensor
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 4. Initialize Model
    model = UniDirectionalLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, max_class + 1, DROPOUT).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    
    print("-" * 30)
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        val_epoch_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {epoch_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # Save Best Model (based on Accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New Best Model Saved! ({val_acc:.2f}%)")
            
    print("-" * 30)
    print("Training Complete.")
    
    # 6. Evaluation on Best Model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    final_preds = []
    final_targets = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            final_preds.extend(predicted.cpu().numpy())
            final_targets.extend(labels.cpu().numpy())
            
    # Confusion Matrix
    cm = confusion_matrix(final_targets, final_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    target_names = [EMOTION_MAP[i] for i in sorted(list(set(final_targets) | set(final_preds)))]
    print("\nClassification Report:")
    # Handle edge case where some classes are not in val set
    try:
        print(classification_report(final_targets, final_preds)) # target_names=target_names
    except:
        print(classification_report(final_targets, final_preds))
        
    # Plot CM
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Validation Set)')
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion Matrix saved to {CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    main()
