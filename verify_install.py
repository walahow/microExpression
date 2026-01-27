import sys
import importlib.util

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    
    if importlib.util.find_spec(module_name) is None:
        print(f"❌ {display_name}: Not installed")
        return False
    else:
        print(f"✅ {display_name}: Installed")
        return True

print("--- Checking Installation ---")

# Check Torch and CUDA
if importlib.util.find_spec("torch") is not None:
    try:
        import torch
        print(f"✅ Torch Version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: Yes")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"❌ CUDA Available: No (Torch is running on CPU)")
    except Exception as e:
        print(f"❌ Torch Error: {e}")
else:
    print("❌ Torch: Not installed")

check_import("torchvision")
check_import("ultralytics", "YOLOv8 (ultralytics)")
check_import("facenet_pytorch", "FaceNet-PyTorch")
check_import("cv2", "OpenCV")
check_import("numpy", "NumPy")
check_import("pandas", "Pandas")
check_import("emotion_recognizer", "Emotion Recognizer (Local)")
check_import("timm", "timm (PyTorch Image Models)")

print("--- Check Complete ---")
