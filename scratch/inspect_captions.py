import json
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')

sample_files = glob.glob("kaggle_dataset_staging/captions/*.json")[:3]
for p in sample_files:
    with open(p, 'r', encoding='utf-8') as f:
        shots = json.load(f)
        print(f"=== {p} ({len(shots)} shots) ===")
        for s in shots[:3]:
            print(f"Shot {s['shot_id']} ({s['start_time']}s - {s['end_time']}s):")
            print(f"  Caption: {s.get('caption', '')[:80]}")
            print(f"  Memory : {s.get('memory', '')[:80]}")
