import numpy as np
import os
import glob

FEATURES_DIR = r"d:\proj\micpression\data\features_v2"

def check_variance():
    files = glob.glob(os.path.join(FEATURES_DIR, "*.npy"))
    if not files:
        print("No files found.")
        return

    print(f"Checking {len(files)} files for temporal variance...")
    
    variances = []
    zero_variance_count = 0
    total_samples = 0
    
    # Check first 100 files
    for f in files[:100]:
        data = np.load(f)
        # Data shape is likely (Time, 520) or (Time, 8) depending on when it was saved vs how it's loaded
        # The training script loads (T, 520) and slices to (T, 8).
        # Let's see what the raw saved shape is.
        
        # We only care about the last 8 columns (emotion scores) if it's 520
        features = data[:, -8:] 
        
        # Calculate variance across time for each of the 8 emotions
        # shape: (8,)
        var = np.var(features, axis=0)
        variances.append(var)
        
        if np.all(var < 1e-6):
            zero_variance_count += 1
            
        total_samples += 1
        
    variances = np.array(variances)
    mean_variance = np.mean(variances, axis=0)
    
    print(f"Files checked: {total_samples}")
    print(f"Files with effectively ZERO temporal variance: {zero_variance_count}")
    print(f"Average Variance per Emotion Channel: {mean_variance}")
    print(f"Max Variance observed: {np.max(variances)}")
    print(f"Sample Shape: {data.shape}")

if __name__ == "__main__":
    check_variance()
