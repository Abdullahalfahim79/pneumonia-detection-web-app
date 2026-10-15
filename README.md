# Pneumonia X-ray Analyzer

Upload a chest X-ray → predictions from CNN, ViT-B/16 and Hybrid CNN-ViT, plus Grad-CAM or Score-CAM heatmaps.

## 1. Get your trained weights from Kaggle
Open your notebook on Kaggle → **Output** tab (right side) → download these 3 files into the `weights/` folder:
- `best_cnn_model.keras`
- `best_vit_pneumonia.pth`
- `best_hybrid_cnn_vit.pth`

## 2. Install and run (Python 3.10 or 3.11)
```
python -m venv venv
venv\Scripts\activate        # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://localhost:8000

## Notes
- Score-CAM is slower (many forward passes). Grad-CAM takes a few seconds on CPU.
- Hybrid heatmap = average of the CNN branch map and the ViT branch map.
- Preprocessing matches the notebook (224x224, CNN input in [0,1], ViT ImageNet-normalized).
- To deploy online, use a host that fits ~400 MB of weights and ~2 GB RAM (Hugging Face Spaces, Render, a small VPS).
