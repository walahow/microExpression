
import numpy as np
import os
import glob
import re
import json
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

FEATURES_DIR = r"d:\proj\micpression\data\features_v2"
OUTPUT_MAP = "subject_map.json"

def infer_subjects():
    print("Loading feature files...")
    files = glob.glob(os.path.join(FEATURES_DIR, "*.npy"))
    if not files:
        print("No feature files found.")
        return

    # 1. Collect one face embedding per video
    video_embeddings = []
    file_ids = []
    filenames = []

    for f in files:
        fname = os.path.basename(f)
        match = re.search(r'(\d+)', fname)
        if not match: continue
        
        vid_id = int(match.group(1))
        
        # Load only the first frame
        try:
            # Load metadata only first? No, npy doesn't support lazy load easily without mmap
            # But specific files are small enough.
            data = np.load(f) # Shape (T, 520)
            
            # Check shape
            if data.shape[1] != 520:
                # If it's already 8, we can't do this
                continue
                
            # Face embedding is the first 512 cols
            face_emb = data[0, :512] 
            
            video_embeddings.append(face_emb)
            file_ids.append(vid_id)
            filenames.append(fname)
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    if not video_embeddings:
        print("Could not extract embeddings. Are files already sliced to 8 dims?")
        return

    X = np.array(video_embeddings)
    print(f"Loaded {len(X)} video embeddings. Shape: {X.shape}")
    
    # 2. Cluster faces
    # DeepFace/FaceNet embeddings are usually normalized.
    # Cosine distance < epsilon. 
    # DBSCAN is good because we don't know exact number of clusters (expecting ~26)
    # Epsilon 0.5 (cosine distance) is usually a loose threshold for faces. 
    # But since these are from the SAME video, they should be identical.
    # Across different videos of same person, should be very close.
    # Let's use cosine similarity. DBSCAN uses euclidean by default.
    # Normalized vectors: Euclidean distance relates to Cosine distance.
    # dist = sqrt(2*(1-cos_sim)). If sim=0.9, dist=sqrt(0.2)=0.44.
    
    # Let's normalize X first
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / (norms + 1e-10)
    
    # Run DBSCAN with parameter sweep
    best_eps = 0.5
    best_diff = 999
    final_labels = []
    
    print("\n--- Parameter Sweep ---")
    for eps in np.arange(0.1, 0.7, 0.05):
        clustering = DBSCAN(eps=eps, min_samples=2, metric='euclidean', n_jobs=-1)
        labels = clustering.fit_predict(X_norm)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        diff = abs(n_clusters - 26)
        print(f"Eps: {eps:.2f} -> Clusters: {n_clusters}, Noise: {n_noise}")
        
        # Penalize over-segmentation (More clusters than 26 = Risk of Leakage)
        # We prefer Under-segmentation (Fewer clusters = Safe)
        if n_clusters > 26:
            score = diff + 100 # Heavy penalty
        else:
            score = diff
            
        if score < best_diff and n_clusters >= 10: 
            best_diff = score
            best_eps = eps
            final_labels = labels
            
    print(f"\nSelected Best Eps: {best_eps:.2f}")
    labels = final_labels
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    print(f"\n--- Final Clustering Results ---")
    print(f"Estimated No. of Subjects: {n_clusters}")
    
    # 3. Create Mapping
    mapping = {}
    for vid_id, cluster_id in zip(file_ids, labels):
        # Assign noise points to their own unique "unknown" subject IDs to be safe
        if cluster_id == -1:
            final_sub_id = 999 + int(vid_id) # Offset to avoid collision
        else:
            final_sub_id = int(cluster_id)
            
        mapping[int(vid_id)] = final_sub_id
        
    # Save
    with open(OUTPUT_MAP, 'w') as f:
        json.dump(mapping, f, indent=4)
        
    print(f"Saved Subject Mapping to {OUTPUT_MAP}")
    
    import collections
    counts = collections.Counter(labels)
    print("\nVideos per Subject (Cluster ID : Count):")
    sorted_counts = dict(sorted(counts.items()))
    for k, v in sorted_counts.items():
        print(f"Subject {k}: {v} videos")

if __name__ == "__main__":
    infer_subjects()
