import requests
import os
import sys

# List of potential URLs for yolov8n-face.pt
URLS = [
    "https://huggingface.co/Bingsu/yolov8-face/resolve/main/yolov8n-face.pt",
    "https://github.com/derronqi/yolov8-face/releases/download/v1.0.0/yolov8n-face.pt",
    "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.pt", # Will rename to yolov8n-face.pt
]

OUTPUT_FILE = "yolov8n-face.pt"

def download_file(url, filename):
    print(f"Trying to download from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                downloaded += len(chunk)
                f.write(chunk)
                # Simple progress bar
                if total_size > 0:
                    percent = int(50 * downloaded / total_size)
                    sys.stdout.write(f"\r[{'=' * percent}{' ' * (50 - percent)}] {downloaded}/{total_size} bytes")
                    sys.stdout.flush()
        
        print(f"\n✅ Success! Saved to {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return False

def main():
    if os.path.exists(OUTPUT_FILE):
        print(f"File {OUTPUT_FILE} already exists. Removing to retry...")
        os.remove(OUTPUT_FILE)
        
    for url in URLS:
        if download_file(url, OUTPUT_FILE):
            print("Download complete. You can now run the pipeline script.")
            return

    print("❌ All download attempts failed.")
    print("Please manually download 'yolov8n-face.pt' and place it in this directory.")

if __name__ == "__main__":
    main()
