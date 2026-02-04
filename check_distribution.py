import os
import glob
import re
from collections import Counter

# --- Configuration ---
# Check which directory exists
if os.path.exists("data/features_celeba"):
    FEATURES_DIR = "data/features_celeba"
    print(f"Checking features in: {FEATURES_DIR}")
elif os.path.exists("data/features_v2"):
    FEATURES_DIR = "data/features_v2"
    print(f"Checking features in: {FEATURES_DIR}")
else:
    print("No feature directory found to check.")
    exit()

def get_file_id(fpath):
     fname = os.path.basename(fpath)
     # Match the logic in train_lstm.py
     match = re.search(r'(\d+)', fname)
     return int(match.group(1)) if match else 0

def get_emotion(fpath):
    fname = os.path.basename(fpath)
    # Filename format: emotion_....npy
    return fname.split('_')[0]

def main():
    all_files = glob.glob(os.path.join(FEATURES_DIR, "*.npy"))
    if len(all_files) == 0:
        print("No files found.")
        return

    # Simulate the split
    all_files.sort(key=get_file_id)
    split_idx = int(0.8 * len(all_files))
    
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    # Analysis
    print(f"\nTotal Files: {len(all_files)}")
    print(f"Train Files: {len(train_files)}")
    print(f"Test Files:  {len(test_files)}")
    
    train_emotions = [get_emotion(f) for f in train_files]
    test_emotions = [get_emotion(f) for f in test_files]
    
    train_counts = Counter(train_emotions)
    test_counts = Counter(test_emotions)
    
    print("\n--- Train Distribution ---")
    for emo, count in train_counts.items():
        pct = (count / len(train_files)) * 100
        print(f"{emo:<15}: {count:3d} ({pct:.1f}%)")
        
    print("\n--- Test Distribution ---")
    for emo, count in test_counts.items():
        pct = (count / len(test_files)) * 100
        print(f"{emo:<15}: {count:3d} ({pct:.1f}%)")
        
    # Check for missing emotions in test
    all_emotions = set(train_counts.keys()) | set(test_counts.keys())
    missing_in_test = all_emotions - set(test_counts.keys())
    
    if missing_in_test:
        print(f"\n[WARNING] The following emotions are MISSING from the Test Set: {missing_in_test}")
    else:
        print("\n[OK] All emotions are present in the Test Set.")

if __name__ == "__main__":
    main()
