import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import os
from tqdm import tqdm
from setup_celeba import setup_celeba_subset

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 5 # Small number for "Proof of Concept"
LEARNING_RATE = 0.001
SUBSET_SIZE = 10000
DATA_DIR = 'data/celeba'
SAVE_PATH = 'celeba_backbone.pth'

def train_celeba_backbone():
    print("--- Starting CelebA Pre-training Experiment ---")
    
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2. Data Preparation
    # Transforms: Resize to standard 224x224 for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Dataset
    from setup_celeba import setup_celeba_subset, CustomCelebADataset
    
    # Get Subset Indices First
    subset_indices = setup_celeba_subset(DATA_DIR, SUBSET_SIZE)
    if subset_indices is None:
        return

    try:
        # Load CUSTOM Dataset with the subset indices directly
        subset_dataset = CustomCelebADataset(
            root_dir=DATA_DIR, 
            transform=transform, 
            subset_indices=subset_indices
        )
    except Exception as e:
        print(f"Dataset init failed: {e}")
        return

    dataloader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    print(f"Training on {len(subset_dataset)} images.")
    
    # 3. Model Setup (ResNet18)
    # We use a standard ResNet18. 
    # num_classes = 40 (CelebA has 40 attributes like 'Smiling', 'Male', etc.)
    model = torchvision.models.resnet18(pretrained=True) 
    # Change last layer for 40 attributes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 40)
    
    model = model.to(device)
    
    # 4. Training
    # Multi-label classification -> BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).float() # BCE expects float inputs
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
        print(f"Epoch {epoch+1} Loss: {running_loss / len(dataloader):.4f}")
        
    # 5. Save the Backbone
    # We want the feature extractor, so we usually remove the fc layer, 
    # but here we save the whole thing and let train_lstm handle the stripping.
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Pre-training complete. Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_celeba_backbone()
