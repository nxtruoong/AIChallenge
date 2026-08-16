import json

with open("data/eval/evaluation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Evaluation Summary:")
print("Total:", len(data))
hits1 = sum(1 for d in data if d.get("rank", 999) == 1)
hits5 = sum(1 for d in data if 1 <= d.get("rank", 999) <= 5)
hits10 = sum(1 for d in data if 1 <= d.get("rank", 999) <= 10)
print(f"Recall@1:  {hits1/len(data)*100:.2f}% ({hits1}/{len(data)})")
print(f"Recall@5:  {hits5/len(data)*100:.2f}% ({hits5}/{len(data)})")
print(f"Recall@10: {hits10/len(data)*100:.2f}% ({hits10}/{len(data)})")
