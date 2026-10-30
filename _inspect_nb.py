import json
from pathlib import Path
p = Path(r"C:\Users\Fast\Downloads\shi (1).ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
out = Path(r"D:\Smog-Hazard-Index\_nb_cells.txt")
parts = []
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if not src.strip():
        continue
    ctype = c["cell_type"]
    parts.append("=" * 80)
    parts.append("CELL %d type=%s" % (i, ctype))
    parts.append("=" * 80)
    parts.append(src)
    parts.append("")
out.write_text("\n".join(parts), encoding="utf-8")
print("wrote", out, "bytes", out.stat().st_size)
