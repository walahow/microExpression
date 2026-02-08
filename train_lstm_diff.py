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
FEATURES_PATH = os.path.join(PROJECT_ROOT, "X_diff.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "y_diff.npy")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "lstm_diff_specialist.pth")
CONFUSION_MATRIX_PATH = os.path.join(PROJECT_ROOT, "confusion_matrix_diff.png")
LOSS_CURVE_PATH = os.path.join(PROJECT_ROOT, "loss_curves_diff.png")

# Hyperparameters
INPUT_SIZE = 1280  # EfficientNet-B0 feature dimension
HIDDEN_SIZE = 128
NUM_LAYERS = 1
NUM_CLASSES = 3  # Others, Disgust, Repression
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Label Mapping (String -> Integer)
LABEL_MAP = {
    'Others': 0,
    'Disgust': 1,
    'Repression': 2
}

# Reverse mapping for display
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMDiffSpecialist(nn.Module):
    """Unidirectional LSTM for micro-expression classification using difference features."""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
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
        
        # Take the last time step
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Classification
        out = self.fc(last_out)  # (batch, num_classes)
        return out


def encode_labels(labels_str):
    """Convert string labels to integer labels."""
    return np.array([LABEL_MAP[label] for label in labels_str])


def compute_class_weights(y_train, num_classes, device):
    """Compute class weights as 1.0 / count."""
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    weights = np.zeros(num_classes)
    for cls in range(num_classes):
        if cls in class_counts:
            weights[cls] = 1.0 / class_counts[cls]
        else:
            weights[cls] = 0.0  # No samples for this class
    
    # Normalize weights
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32).to(device)


def main():
    device = get_device()
    print(f"Training on device: {device}")
    
    # 1. Load Data
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(LABELS_PATH):
        print(f"ERROR: Data files not found!")
        print(f"  Expected: {FEATURES_PATH}")
        print(f"  Expected: {LABELS_PATH}")
        print(f"\nPlease run extract_features_diff.py first.")
        return
    
    X = np.load(FEATURES_PATH)
    y_str = np.load(LABELS_PATH)
    
    print(f"\nLoaded Data:")
    print(f"  Features: {X.shape}")
    print(f"  Labels: {y_str.shape}")
    
    # 2. Encode labels to integers
    y = encode_labels(y_str)
    
    print(f"\nOriginal class distribution:")
    unique_str, counts_str = np.unique(y_str, return_counts=True)
    for label, count in zip(unique_str, counts_str):
        print(f"  {label}: {count}")
    
    # 3. Train/Test Split (Stratified)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\nStratified split successful")
    except ValueError as e:
        print(f"\nWARNING: Stratified split failed ({e}). Using random split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Val set:   {X_val.shape[0]} samples")
    
    # 4. Compute Class Weights
    class_weights = compute_class_weights(y_train, NUM_CLASSES, device)
    print(f"\nClass weights (normalized): {class_weights.cpu().numpy()}")
    
    # 5. Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 6. Initialize Model
    model = LSTMDiffSpecialist(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 7. Training Loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*70}")
    print(f"Starting Training ({NUM_EPOCHS} epochs)...")
    print(f"{'='*70}")
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
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
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        
        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] "
              f"Loss: {epoch_loss:.4f} Acc: {train_acc:5.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} Val Acc: {val_acc:5.2f}%", end="")
        
        # Save best model based on validation loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(" ← Best model saved!")
        else:
            print()
    
    print(f"{'='*70}")
    print(f"Training Complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"{'='*70}")
    
    # 8. Plot Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(val_losses, label='Val Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)
    print(f"\nLoss curves saved to: {LOSS_CURVE_PATH}")
    
    # 9. Evaluation on Best Model
    print(f"\nEvaluating best model on validation set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    target_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=target_names))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Validation Set)')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    
    print(f"\n{'='*70}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
