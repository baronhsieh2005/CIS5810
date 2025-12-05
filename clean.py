from pathlib import Path

base = Path("runs/pose/predict3")
img_dir = base                    # images are directly in predict3
label_dir = base / "labels"

# collect image stems (filename without extension)
img_stems = {p.stem for p in img_dir.glob("*") if p.is_file()}

for lbl in label_dir.glob("*.txt"):
    if lbl.stem not in img_stems:
        print("Deleting orphan label:", lbl)
        lbl.unlink()