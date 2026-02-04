
import os
import glob
import re

FEATURES_DIR = r"d:\proj\micpression\data\features_v2"

def verify_leakage():
    files = glob.glob(os.path.join(FEATURES_DIR, "*.npy"))
    if not files:
        print("No features found.")
        return

    ids = []
    for f in files:
        fname = os.path.basename(f)
        # Regex from train_lstm.py
        match = re.search(r'(\d+)', fname)
        if match:
            ids.append(int(match.group(1)))
            
    ids = sorted(list(set(ids)))
    print(f"Total Unique IDs found: {len(ids)}")
    print(f"Min ID: {min(ids)}")
    print(f"Max ID: {max(ids)}")
    
    if len(ids) > 30:
        print("\nCRITICAL WARNING: CASME II has only ~26 subjects.")
        print(f"Finding {len(ids)} unique IDs implies these are VIDEO IDs, not SUBJECT IDs.")
        print("Splitting by Video ID allows the same subject to be in both Train and Test sets.")
        print("CONCLUSION: DATA LEAKAGE DETECTED.")
    else:
        print("\nIDs count represents plausible subject count.")

if __name__ == "__main__":
    verify_leakage()
