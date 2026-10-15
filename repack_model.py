import os
import zipfile

folder = os.path.join("weights", "best_cnn_model")
output = os.path.join("weights", "best_cnn_model.keras")

required = ["config.json", "metadata.json", "model.weights.h5"]

for name in required:
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
    for name in required:
        z.write(os.path.join(folder, name), arcname=name)

print("Created:", os.path.abspath(output))
