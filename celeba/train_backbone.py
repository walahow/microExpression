import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "celeba")
IMG_DIR = os.path.join(DATA_ROOT, "img_align_celeba", "img_align_celeba") # Check nesting
ATTR_FILE = os.path.join(DATA_ROOT, "list_attr_celeba.csv")

# Weights from Stage 1 (FFHQ)
FFHQ_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights_ffhq_pretrained.pth")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "weights_celeba_finetuned.pth")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 # Lower LR for fine-tuning
NUM_EPOCHS = 15
IMAGE_SIZE = 224

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
class CelebAClassifier(nn.Module):
    def __init__(self, num_classes=40, pretrained_path=None):
        super(CelebAClassifier, self).__init__()
        
        # 1. Define Backbone (Same structure as FFHQ Encoder)
        # ResNet18 without the final FC layer
        base_model = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1]) # Output: [B, 512, 1, 1]
        
        # 2. Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
        # 3. Load Pretrained Weights (if available)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading FFHQ pretrained weights from {pretrained_path}...")
            try:
                state_dict = torch.load(pretrained_path, map_location=get_device())
                # The FFHQ encoder was saved as a Sequential model directly
                # So the keys should match self.backbone exactly
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            except Exception as e:
                print(f"ERROR loading weights: {e}")
        else:
            print("WARNING: No pretrained weights found. Training from scratch!")

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

# --- Dataset ---
class CelebADataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, limit=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Read CSV
        # CelebA CSV usually has image_id as first column, then -1/1 for attributes
        self.df = pd.read_csv(csv_file)
        
        # Handle filename column (sometimes index, sometimes 'image_id')
        if 'image_id' in self.df.columns:
            self.filenames = self.df['image_id'].values
            self.labels = self.df.drop('image_id', axis=1).values
        else:
            # Assume first column is filename if no header match
            self.filenames = self.df.iloc[:, 0].values
            self.labels = self.df.iloc[:, 1:].values
            
        # Convert -1 to 0 for BCE Loss
        self.labels = (self.labels + 1) // 2 
        
        if limit:
            self.filenames = self.filenames[:limit]
            self.labels = self.labels[:limit]
            print(f"DEBUG: Limited dataset to {limit} samples")

        print(f"Loaded {len(self.filenames)} samples. {self.labels.shape[1]} Attributes.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        # Handle potential nesting issues or exact filename
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # Placeholder for missing images
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# --- Training Loop ---
def train():
    device = get_device()
    print(f"Device: {device}")
    
    # 1. Setup Data
    if not os.path.exists(IMG_DIR):
        # Fallback check for un-nested structure
        alt_img_dir = os.path.join(DATA_ROOT, "img_align_celeba")
        if os.path.exists(os.path.join(alt_img_dir, "000001.jpg")):
             IMG_DIR_FINAL = alt_img_dir
        else:
             print(f"ERROR: Could not find images in {IMG_DIR} or {alt_img_dir}")
             return
    else:
        IMG_DIR_FINAL = IMG_DIR

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use subset for smoke test? Or full data?
    # Let's verify dataset presence first
    dataset = CelebADataset(ATTR_FILE, IMG_DIR_FINAL, transform=train_transform)
    
    # Split? CelebA has specific partitions but for now simple random split or full train is okay
    # For this backbone trainer, we usually use the train partition, but using all is fine for feature power
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 2. Model
    model = CelebAClassifier(num_classes=40, pretrained_path=FFHQ_WEIGHTS_PATH).to(device)
    
    # 3. Optimization
    criterion = nn.BCEWithLogitsLoss() # Multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Loop
    print(f"Starting Fine-tuning on CelebA for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {running_loss/len(dataloader):.4f}")
        
        # Save checkpoints
        torch.save(model.backbone.state_dict(), SAVE_PATH) # Save ONLY the backbone for Stage 3
        print(f"Saved backbone weights to {SAVE_PATH}")

if __name__ == "__main__":
    train()
