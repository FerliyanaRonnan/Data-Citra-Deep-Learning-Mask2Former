import json
from pathlib import Path

def ipynb_to_txt(ipynb_path):
    ipynb_path = Path(ipynb_path)
    output_path = ipynb_path.with_suffix(".txt")

    # Baca file ipynb
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    lines = []

    # Ambil isi semua cell
    for i, cell in enumerate(notebook["cells"], start=1):
        cell_type = cell["cell_type"]

        lines.append(f"\n{'='*50}")
        lines.append(f"CELL {i} ({cell_type.upper()})")
        lines.append(f"{'='*50}\n")

        source = "".join(cell.get("source", []))
        lines.append(source)

    # Simpan jadi txt
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Berhasil convert ke: {output_path}")


# Ganti sesuai nama file kamu
ipynb_to_txt("Mask2Form (512 x 512).ipynb")