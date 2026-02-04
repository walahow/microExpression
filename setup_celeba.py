import os
import torch
import torchvision
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomCelebADataset(Dataset):
    """
    Custom Dataset to load CelebA from the Kaggle/Manual structure (Attributes in CSV/TXT)
    """
    def __init__(self, root_dir, transform=None, subset_indices=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.transform = transform
        
        # Detect Attribute File (TXT or CSV)
        csv_path = os.path.join(root_dir, 'list_attr_celeba.csv')
        txt_path = os.path.join(root_dir, 'list_attr_celeba.txt')
        
        if os.path.exists(csv_path):
            self.attr_df = pd.read_csv(csv_path)
            self.image_names = self.attr_df['image_id'].values
            # Drop image_id column to get attributes only
            self.attrs = self.attr_df.drop('image_id', axis=1).values
        elif os.path.exists(txt_path):
            # Read standard CelebA .txt
            self.attr_df = pd.read_csv(txt_path, delim_whitespace=True, header=1)
            self.image_names = self.attr_df.index.values
            self.attrs = self.attr_df.values
        else:
            raise RuntimeError("Could not find list_attr_celeba.csv or .txt")
            
        # Replace -1 with 0 for binary classification
        self.attrs[self.attrs == -1] = 0
        
        # Handle Subset
        if subset_indices is not None:
            self.image_names = self.image_names[subset_indices]
            self.attrs = self.attrs[subset_indices]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        attr = torch.tensor(self.attrs[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, attr

def setup_celeba_subset(root_dir='data/celeba', subset_size=10000):
   # ... (Old logic replaced by dataset class availability)
   # Just verifying file existence here
   img_dir = os.path.join(root_dir, 'img_align_celeba')
   if not os.path.exists(img_dir):
       print("Image directory not found.")
       return None
       
   # Creates random indices
   # We need total length first.
   try:
       # Initialize temp to get length
       ds = CustomCelebADataset(root_dir)
       total_len = len(ds)
       print(f"Found {total_len} images.")
       
       indices = list(range(total_len))
       import numpy as np
       np.random.shuffle(indices)
       return indices[:subset_size]
   except Exception as e:
       print(f"Error setting up subset: {e}")
       return None


if __name__ == "__main__":
    setup_celeba_subset()
