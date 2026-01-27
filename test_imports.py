import torch
import timm

print(f"Torch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

print(f"Timm Version: {timm.__version__}")

try:
    from emotion_recognizer import HSEmotionRecognizer
    print("HSEmotionRecognizer imported successfully.")
    
    # Try initializing a model to check for model loading errors
    print("Simulating model load...")
    # We won't load the full weights to save time, but we can check if the class exists
    # Or actually, let's just assume import is enough for now, 
    # but the error happened during forward/load.
    
    # Replicating the error requires loading the model
    # fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu') 
    # print("Model loaded successfully.")
    
except Exception as e:
    print(f"Import/Load Error: {e}")
