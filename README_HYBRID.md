# Hybrid Macro-Micro Expression Detection - Quick Start

## Overview
Real-time webcam application combining:
- **HSEmotion** (Macro): 8 classes (Anger, Happiness, Neutral, etc.)
- **LSTM Specialist** (Micro): 3 classes (Disgust, Repression, Others)

## Running the App

```bash
python main_final.py
```

**Controls**:
- `q` - Quit
- `r` - Reset buffer

## How It Works

### Fusion Logic (Priority-Based)

1. **High-Confidence Macro** (>75%)
   - Display: `MACRO: {EXPRESSION}` (Green)
   - Example: Strong smile → "MACRO: HAPPINESS"

2. **Neutral/Low-Confidence → Check Micro**
   - **Disgust**: "MICRO-DISGUST (Hidden)" (Red)
   - **Repression**: "REPRESSION (Suppressed)" (Orange)
   - **Others**: "MICRO-MOVEMENT" (Yellow)

3. **Default**: "NEUTRAL" (Grey)

### Example Scenarios

**Strong Emotion**:
```
Macro confidence: 92% (Happiness)
→ Display: "MACRO: HAPPINESS" (Green)
```

**Hidden Micro-Expression**:
```
Macro: Neutral (68%)
Micro: Disgust
→ Display: "MICRO-DISGUST (Hidden)" (Red)
```

## UI Elements

- **Main Status** (Top-left): Current detection
- **Bounding Box**: Color-coded around face
- **Buffer Bar**: Shows initialization progress (15 frames needed)
- **Debug Info** (Bottom): Raw macro/micro predictions
- **FPS** (Top-right): Performance metric

## Requirements

- Webcam
- CUDA GPU (recommended)
- Required files:
  - `enet_b0_8_best_vgaf.pt` (HSEmotion)
  - `lstm_diff_specialist.pth` (LSTM)
  - `yolov8n-face.pt` (Face detection)

## Troubleshooting

**Issue**: WinError 1337  
**Solution**: Already fixed with monkeypatch in the script

**Issue**: Low FPS (<15)  
**Solution**: Use GPU, reduce camera resolution

**Issue**: "INITIALIZING..." stuck  
**Solution**: Ensure face is visible and well-lit

## Technical Details

- **Buffer**: 15 frames @ 30 FPS = 0.5 seconds
- **Features**: 1280-dim from EfficientNet-B0 backbone
- **Difference**: D_i = F_i - F_0 (removes identity/head movement)
- **LSTM**: Unidirectional, trained on 188 CASME II samples

For full documentation, see `hybrid_system_guide.md`.
