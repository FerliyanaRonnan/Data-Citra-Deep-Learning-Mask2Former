import json

with open("Mask2Form-Pseudo.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = "".join(cell['source'])
    new = src.replace(
        "    'best_miou': trainer.best_score,",
        "    'best_score': trainer.best_score,"
    )
    if new != src:
        cell['source'] = new.splitlines(keepends=True)
        changes.append(i)

print(f"Fixed cells: {changes}")
with open("Mask2Form-Pseudo.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Done")
