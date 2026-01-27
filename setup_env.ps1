# Setup Script for Facial Expression Recognition System
# This script installs the required libraries with CUDA support into the 'venv' virtual environment.

Write-Host "Starting installation..."

# Ensure venv exists
if (-not (Test-Path "venv")) {
    Write-Host "Virtual environment 'venv' not found. Creating it..."
    python -m venv venv
}

# Upgrade pip
Write-Host "Upgrading pip..."
.\venv\Scripts\python.exe -m pip install --upgrade pip

# FORCE UNINSTALL existing torch if it's the wrong version (e.g. CPU version)
Write-Host "Uninstalling any existing PyTorch versions to ensure clean CUDA install..."
.\venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA 12.1 support
# GTX 1650 supports CUDA 12.x. We use the official PyTorch index for cu121.
Write-Host "Installing PyTorch (torch, torchvision) with CUDA 12.1 support..."
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
# ultralytics (YOLOv8), facenet-pytorch, opencv, numpy, pandas
Write-Host "Installing ultralytics, facenet-pytorch, and other utilities..."
.\venv\Scripts\python.exe -m pip install ultralytics facenet-pytorch opencv-python numpy pandas

Write-Host "Installation complete!"
Write-Host "To activate the environment, run: .\venv\Scripts\Activate.ps1"
