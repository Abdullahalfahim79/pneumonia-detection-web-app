
import os
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

WEIGHTS = os.path.join(os.path.dirname(__file__), "weights")

# ---------- Verify ViT ----------
vit_path = os.path.join(WEIGHTS, "best_vit_pneumonia.pth")
print("\n--- ViT checkpoint ---")

state = torch.load(vit_path, map_location="cpu", weights_only=True)

vit = vit_b_16(weights=None)
vit.heads.head = nn.Linear(vit.heads.head.in_features, 2)

print("Checkpoint head keys:")
for key in state:
    if key.startswith("heads."):
        print(key, tuple(state[key].shape))

try:
    result = vit.load_state_dict(state, strict=True)
    print("ViT architecture match: SUCCESS")
    print(result)
except Exception as exc:
    print("ViT architecture match: FAILED")
    print(exc)

# ---------- Verify Hybrid classifier ----------
hybrid_path = os.path.join(WEIGHTS, "best_hybrid_cnn_vit.pth")
print("\n--- Hybrid checkpoint ---")

hybrid_state = torch.load(
    hybrid_path, map_location="cpu", weights_only=True
)

class HybridCNNViTClassifier(nn.Module):
    def __init__(self, input_dim=896):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(x)

hybrid = HybridCNNViTClassifier(input_dim=896)

try:
    result = hybrid.load_state_dict(hybrid_state, strict=True)
    print("Hybrid architecture match: SUCCESS")
    print(result)

    hybrid.eval()
    with torch.no_grad():
        sample = torch.randn(1, 896)
        output = hybrid(sample)
    print("Hybrid test output shape:", tuple(output.shape))
except Exception as exc:
    print("Hybrid architecture match: FAILED")
    print(exc)
