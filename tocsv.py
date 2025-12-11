import csv
from pathlib import Path

root = Path("./dead_lift_variant/images")
out_csv = root / "deadlift_phase_labels.csv"

rows = []
for phase_label, subdir in [("lowering", "lowering"), ("lifting", "lifting"), ("lockup", "lockup")]:
    for img_path in sorted((root / subdir).glob("*.jpg")):
        parts = img_path.stem.split("-")
        # Optional: infer video_id / frame_idx from filename if you follow a convention
        video_id = parts[1]     # e.g. bench1_frame0001 -> bench1
        frame_idx = parts[-1]   # e.g. bench1_frame0001 -> frame0001
        rows.append({
            "video_id": video_id,
            "frame_idx": frame_idx,
            "img_path": str(img_path),
            "phase_label": phase_label,
        })


fieldnames = ["video_id", "frame_idx", "img_path", "phase_label"]
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Wrote", out_csv)