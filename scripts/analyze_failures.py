import json
from pathlib import Path

data = json.load(open('data/manuals/langchain_baseline/recall_eval_results.json', encoding='utf-8'))
for failure in data['summary']['failures']:
    print(f"[{failure['manual']}] {failure['query']} (Needs: {failure['must_contain']})")
