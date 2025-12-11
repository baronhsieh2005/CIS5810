from pathlib import Path

# 1) UPDATE THESE PATHS to match your setup
original_dir = Path("./dead_lift_variant/images")
lifting_dir    = Path("./dead_lift_variant/custom_dataset/lifting")
lockup_dir  = Path("./dead_lift_variant/custom_dataset/lockup")
lowering_dir = Path("./dead_lift_variant/custom_dataset/lowering")

# 2) Build a set of filenames to KEEP (from train + val)
keep_names = set()

lift = 0
lockup = 0
lower = 0

for p in lifting_dir.glob("*.jpg"):
    keep_names.add(p.name)
    lift += 1

for p in lockup_dir.glob("*.jpg"):
    keep_names.add(p.name)
    lockup += 1

for p in lowering_dir.glob("*.jpg"):
    keep_names.add(p.name)
    lower += 1

print("Lift:", lift)
print("Lock:", lockup)
print("Lower:", lower)


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
