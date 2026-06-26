import json

with open(r'C:\Users\Ababil Khoerul Imam\Downloads\Mask2Form_Fixed2.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

for i in range(12, 24):
    src = ''.join(cells[i]['source'])
    print(f'=== CELL {i} (items={len(cells[i]["source"])}) ===')
    print(src[:600])
    print('...\n')
