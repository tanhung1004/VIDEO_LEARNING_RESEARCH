from pathlib import Path
import pandas as pd
from PIL import Image

summary = Path("module1/results/ocr_best_integration/keyframe_summary.csv")
df = pd.read_csv(summary)

bad = []

for _, r in df.iterrows():
    video = str(r["video_id"]).strip()
    ts = int(r["timestamp_sec"])
    expected = Path(str(r["image_file"]).strip())

    # Thi?u file
    if not expected.exists():
        bad.append((video, ts, "MISSING", str(expected)))
        continue

    # File t?n t?i nhung l?i/h?ng
    try:
        if expected.stat().st_size < 5000:
            bad.append((video, ts, "TOO_SMALL", str(expected)))
            continue

        with Image.open(expected) as img:
            img.verify()

    except Exception as e:
        bad.append((video, ts, f"INVALID: {e}", str(expected)))

print("=" * 80)
print(f"TOTAL expected frames: {len(df)}")
print(f"BAD / MISSING frames: {len(bad)}")
print("=" * 80)

for x in bad:
    print(x)
