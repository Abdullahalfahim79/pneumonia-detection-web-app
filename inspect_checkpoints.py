import zipfile
import os

files = [
    r"E:\OneDrive\Desktop\Project Development\best_vit_pneumonia.zip",
    r"E:\OneDrive\Desktop\Project Development\best_hybrid_cnn_vit.zip",
]

for path in files:
    print("\n" + "=" * 65)
    print("ZIP:", os.path.basename(path))

    if not os.path.isfile(path):
        print("ERROR: File not found")
        continue

    try:
        with zipfile.ZipFile(path, "r") as z:
            entries = z.infolist()
            if not entries:
                print("ZIP is empty")
            for item in entries:
                print(f"{item.filename}  |  {item.file_size:,} bytes")
    except zipfile.BadZipFile:
        print("ERROR: This is not a valid ZIP file")
