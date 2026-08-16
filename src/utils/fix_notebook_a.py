import json
import io

with open("kaggle_notebook_a_preprocessing.ipynb", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1
bad_str_1 = '"meta_rows = []\n    dropped_count = 0\\n",'
good_str_1 = '"meta_rows = []\\n",\n    "dropped_count = 0\\n",'
content = content.replace(bad_str_1, good_str_1)

# Fix 2
bad_str_2 = '"print(\\"PREPROCESSING COMPLETE — ARTIFACT SUMMARY:\\")\n    print(\\"=\\"*60)\\n",'
good_str_2 = '"print(\\"PREPROCESSING COMPLETE — ARTIFACT SUMMARY:\\")\\n",\n    "print(\\"=\\"*60)\\n",'
content = content.replace(bad_str_2, good_str_2)

with open("kaggle_notebook_a_preprocessing.ipynb", "w", encoding="utf-8") as f:
    f.write(content)

# Test load
try:
    with open("kaggle_notebook_a_preprocessing.ipynb", "r", encoding="utf-8") as f:
        json.load(f)
    print("Notebook A JSON is now valid.")
except Exception as e:
    print(f"Error parsing Notebook A: {e}")
