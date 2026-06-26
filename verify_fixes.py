import json

with open("Mask2Form-Pseudo.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

issues = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = "".join(cell['source'])
    if "self.best_miou" in src:
        issues.append(f"Cell {i}: self.best_miou still exists")
    if "'best_miou': trainer" in src:
        issues.append(f"Cell {i}: best_miou checkpoint key still exists")
    if "scheduler.load_state_dict(checkpoint" in src:
        issues.append(f"Cell {i}: scheduler.load_state_dict still exists")

if issues:
    for x in issues:
        print(x)
else:
    print("CLEAN: All 3 issues fully resolved across all cells.")
