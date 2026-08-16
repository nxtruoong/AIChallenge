import json
import sys

def show(fpath):
    try:
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        print(f'--- {fpath} ---')
        for i, c in enumerate(data.get('cells', [])):
            print(f'Cell {i} ({c["cell_type"]})')
            src = c.get('source', [])
            if src:
                for line in src[:3]: 
                    print(f'  {line.strip()}')
            else: 
                print('  [Empty]')
    except Exception as e:
        print(f"Error reading {fpath}: {e}")

show('notebooks/notebook_example.ipynb')
show('notebooks/kaggle_generate_vectors.ipynb')
