from pathlib import Path

# 1) UPDATE THESE PATHS to match your setup
original_dir = Path("./images")
train_dir    = Path("./custom_dataset/train")
val_dir      = Path("./custom_dataset/val")

lower_dir = Path("./images/lowering")
push_dir = Path("./images/pushing")

lower_count = 0
push_count = 0

for p in lower_dir.glob("*.jpg"):
    lower_count += 1

for p in push_dir.glob("*.jpg"):
    push_count += 1

print("Push:", push_count)
print("Lower:", lower_count)


res_dir = Path("./runs/pose/predict3")

# 2) Build a set of filenames to KEEP (from train + val)
keep_names = set()

for p in train_dir.glob("*.jpg"):
    keep_names.add(p.name)

for p in val_dir.glob("*.jpg"):
    keep_names.add(p.name)

print("Number of kept filenames:", len(keep_names))
list(sorted(list(keep_names))[:10])


to_remove = []

for img_path in original_dir.glob("*.jpg"):
    if img_path.name not in keep_names:
        to_remove.append(img_path)

print("Original images total:", len(list(original_dir.glob("*.jpg"))))
print("Would remove:", len(to_remove))
print("Example to remove:")
for p in to_remove[:10]:
    print("  ", p.name)


import shutil

discard_dir = original_dir.parent / "raw_frames_discarded"
discard_dir.mkdir(exist_ok=True)

for img_path in to_remove:
    dst = discard_dir / img_path.name
    shutil.move(str(img_path), str(dst))

print("Moved", len(to_remove), "files to", discard_dir)
