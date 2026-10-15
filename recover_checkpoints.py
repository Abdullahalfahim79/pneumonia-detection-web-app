
import os
import shutil
import torch

BASE = r"E:\OneDrive\Desktop\Project Development"
OUT = r"E:\OneDrive\Desktop\Project Development\pneumonia-app\weights"

os.makedirs(OUT, exist_ok=True)

jobs = [
    (
        os.path.join(BASE, "best_vit_pneumonia.zip"),
        os.path.join(OUT, "best_vit_pneumonia.pth"),
    ),
    (
        os.path.join(BASE, "best_hybrid_cnn_vit.zip"),
        os.path.join(OUT, "best_hybrid_cnn_vit.pth"),
    ),
]

for source, destination in jobs:
    print("\n" + "=" * 60)
    print("Checking:", os.path.basename(source))

    if not os.path.isfile(source):
        print("ERROR: Original ZIP not found.")
        continue

    try:
        # Preserve the original PyTorch archive structure.
        shutil.copy2(source, destination)

        print("Copied to:", destination)

        # Validate that PyTorch can read it.
        checkpoint = torch.load(
            destination,
            map_location="cpu",
            weights_only=True
        )

        print("PyTorch load: SUCCESS")
        print("Object type:", type(checkpoint).__name__)

        if isinstance(checkpoint, dict):
            print("Top-level keys:", list(checkpoint.keys())[:25])

            state = checkpoint
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state = checkpoint[key]
                    print("Nested state dictionary:", key)
                    break

            if state and all(
                isinstance(v, torch.Tensor) for v in state.values()
            ):
                print("Tensor parameter count:", len(state))
                print("Sample parameter names:")
                for name in list(state.keys())[:12]:
                    print(" ", name, tuple(state[name].shape))

    except Exception as exc:
        print("VALIDATION FAILED")
        print(type(exc).__name__ + ":", exc)

print("\nCheckpoint inspection finished.")
