import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data/eval/evaluation_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
hits = [r for r in results if r['match_rank'] != -1]
misses = [r for r in results if r['match_rank'] == -1]

print(f"Total: {len(results)}, Hits: {len(hits)}, Misses: {len(misses)}")
print("\n--- HITS ---")
for h in hits:
    print(f"Rank {h['match_rank']}: Query: {h['query']} | True: {h['true_video']} @ {h['true_pts']:.1f}s")

print("\n--- MISSES (First 5) ---")
for m in misses[:5]:
    print(f"\nQuery: {m['query']}")
    print(f"True Target: {m['true_video']} @ {m['true_pts']:.1f}s")
    print("Retrieved Top 5:")
    for i, c in enumerate(m['top_10_clips'][:5]):
        print(f"  {i+1}. {c['video_name']} ({c['start_time']:.1f}s - {c['end_time']:.1f}s) score={c['clip_score']:.4f}")
