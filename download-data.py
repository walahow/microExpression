import kagglehub
import shutil
import os

# Define target directory
# We want it in data/CASME2 inside the project directory
target_dir = r"D:\proj\micpression\data\CASME2"

print("Downloading dataset...")
# Download latest version (stored in cache)
path = kagglehub.dataset_download("muhammadzamancuiisb/casme2-preprocessed-v2")
print("Cached path:", path)

# Move/Copy to data folder
print(f"Copying data to {target_dir}...")

# Clean up existing folder if needed
if os.path.exists(target_dir):
    print(f"Target directory {target_dir} already exists. Removing old data...")
    shutil.rmtree(target_dir)

# Copy the entire directory tree
shutil.copytree(path, target_dir)
print("Dataset successfully saved to:", target_dir)