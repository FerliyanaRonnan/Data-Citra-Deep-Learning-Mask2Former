import json
import re

with open("Mask2Form-Pseudo.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = "".join(cell['source'])
    if "scheduler.load_state_dict(checkpoint" not in src:
        continue

    # Remove the scheduler.load_state_dict line (commented or uncommented) and add fast-forward
    # Strategy: find the optimizer.load_state_dict block in try/except and inject fast-forward after it
    
    # Pattern: remove commented-out line if it exists
    src = re.sub(
        r"[ \t]*# scheduler\.load_state_dict\(checkpoint\['scheduler_state_dict'\]\).*?\n",
        "",
        src
    )
    # Pattern: remove real call if it exists
    src = re.sub(
        r"[ \t]*scheduler\.load_state_dict\(checkpoint\['scheduler_state_dict'\]\)\n",
        "",
        src
    )

    cell['source'] = src.splitlines(keepends=True)
    changes.append(i)

print(f"Cleaned scheduler.load_state_dict from cells: {changes}")
with open("Mask2Form-Pseudo.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Done")
