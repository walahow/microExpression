import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "FFHQ")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "weights_ffhq_pretrained.pth")

# --- Hardware Config (Optimized for RTX 3060 / 6GB VRAM) ---
BATCH_SIZE = 48 # Increased to 48 for RTX 3060
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
IMAGE_SIZE = 224
latent_dim = 512
NUM_WORKERS = 4 
SUBSET_SIZE = 52000 # 100% of the dataset

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Autoencoder Architecture ---
class ResNetAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(ResNetAutoencoder, self).__init__()
        
        # Encoder: ResNet18
        # We perform transfer learning by verifying we can reconstruct faces
        # This forces the ResNet to learn good facial features in the bottleneck
        resnet = models.resnet18(pretrained=False) # Start from scratch (or ImageNet if preferred, but we want FFHQ init)
        
        # Remove the final Classification Layer (fc)
        # ResNet18 structure:
        # (conv1): Conv2d(3, 64, ...) -> 112x112
        # (bn1, relu, maxpool) -> 56x56
        # (layer1): 64 ch -> 56x56
        # (layer2): 128 ch -> 28x28
        # (layer3): 256 ch -> 14x14
        # (layer4): 512 ch -> 7x7
        # (avgpool): -> 1x1
        
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]) # Output: [B, 512, 1, 1]
        
        # Decoder: Reconstruct 224x224 image from 512 vector
        self.decoder = nn.Sequential(
            # Input: [B, 512, 1, 1]
            nn.ConvTranspose2d(512, 256, kernel_size=7, stride=1, padding=0), # -> [B, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> [B, 128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [B, 64, 28, 28]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> [B, 32, 56, 56]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # -> [B, 16, 112, 112]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),    # -> [B, 3, 224, 224]
            nn.Sigmoid() # Output pixels 0-1
        )

    def forward(self, x):
        latent = self.encoder(x) # [B, 512, 1, 1]
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# --- Dataset ---
class FFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Recursive search for png/jpg
        valid_exts = ['*.png', '*.jpg', '*.jpeg']
        self.files = []
        for ext in valid_exts:
            self.files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
            
        if SUBSET_SIZE is not None:
             print(f"DEBUG MODE: Limiting dataset to first {SUBSET_SIZE} images.")
             self.files = self.files[:SUBSET_SIZE]

        print(f"Found {len(self.files)} images in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))

def train():
    device = get_device()
    print(f"Device: {device}")
    
    # 1. Prepare Data
    if not os.path.exists(DATA_ROOT):
        print(f"ERROR: FFHQ dataset not found at {DATA_ROOT}")
        print("Please download it from https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq")
        return

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), # 0-1
    ])
    
    dataset = FFHQDataset(DATA_ROOT, transform=transform)
    if len(dataset) == 0:
        print("No images found.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 2. Model
    model = ResNetAutoencoder(latent_dim=latent_dim).to(device)
    
    # 3. Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting Pre-training (Autoencoder)...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for imgs in progress_bar:
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(imgs)
            loss = criterion(reconstructed, imgs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {avg_loss:.6f}")
        
        # Save sample visualization
        with torch.no_grad():
            model.eval()
            # Get a small batch
            try:
                sample_imgs = next(iter(dataloader)).to(device)
                sample_imgs = sample_imgs[:8] # Take top 8
                recon_imgs, _ = model(sample_imgs)
                
                # Concatenate: Top row = Original, Bottom row = Reconstructed
                comparison = torch.cat([sample_imgs, recon_imgs], dim=0)
                
                # Save
                from torchvision.utils import save_image
                save_path = f"reconstruction_epoch_{epoch+1}.png"
                save_image(comparison, save_path, nrow=8)
                print(f"Saved visualization to {save_path}")
            except Exception as e:
                print(f"Could not save visualization: {e}")

        # Save every epoch
        torch.save(model.encoder.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved encoder weights to {MODEL_SAVE_PATH}")

    print("Pre-training Complete.")

if __name__ == "__main__":
    train()
