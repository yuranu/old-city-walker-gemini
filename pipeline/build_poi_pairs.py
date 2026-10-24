#!/usr/bin/env python3
"""
Builds side-by-side `REAL PHOTO | GAME SCREENSHOT` pairs (`pair_<id>.jpg`)
and 3 category contact sheets (`contact_gates.jpg`, `contact_temples.jpg`, `contact_touristic.jpg`)
in `build/qa/cycle1_pois/` for Improvement Cycle 1 (POIs).
"""

import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
PLACES_PATH = ROOT_DIR / "public" / "data" / "places.json"
REF_DIR = ROOT_DIR / "public" / "references"


def make_pair(ref_path: Path, model_path: Path, out_path: Path, title: str) -> None:
    if not model_path.exists():
        return
    m_im = Image.open(model_path).convert("RGB").resize((640, 400), Image.Resampling.LANCZOS)
    if ref_path.exists():
        r_im = Image.open(ref_path).convert("RGB").resize((640, 400), Image.Resampling.LANCZOS)
    else:
        r_im = Image.new("RGB", (640, 400), (42, 42, 48))

    canvas = Image.new("RGB", (1280, 436), (18, 18, 22))
    canvas.paste(r_im, (0, 36))
    canvas.paste(m_im, (640, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), f"REAL REFERENCE — {title}", fill=(235, 220, 185))
    draw.text((656, 10), f"3D GAME MODEL — {title}", fill=(185, 235, 200))
    canvas.save(out_path, quality=85, optimize=True)


def make_contact_sheet(img_paths, out_path: Path, cols: int = 3) -> None:
    valid = [p for p in img_paths if p.exists()]
    if not valid:
        return
    rows = (len(valid) + cols - 1) // cols
    tw, th = 420, 265
    sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    for idx, p in enumerate(valid):
        r = idx // cols
        c = idx % cols
        im = Image.open(p).convert("RGB").resize((tw - 4, th - 24), Image.Resampling.LANCZOS)
        sheet.paste(im, (c * tw + 2, r * th + 22))
        draw.text((c * tw + 8, r * th + 4), p.stem, fill=(240, 230, 200))
    sheet.save(out_path, quality=84, optimize=True)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT_DIR / "build" / "qa" / "cycle1_pois")
    places = json.loads(PLACES_PATH.read_text(encoding="utf-8"))
    for pl in places:
        pid = pl["id"]
        model_jpg = out_dir / f"{pid}.jpg"
        ref_jpg = REF_DIR / f"{pid}.jpg"
        if model_jpg.exists():
            make_pair(ref_jpg, model_jpg, out_dir / f"pair_{pid}.jpg", pl["name"])

    gate_ids = ["jaffa_gate_portal", "new_gate_portal", "damascus_gate_plaza", "herods_gate_portal", "st_anne_and_lions_gate", "golden_gate_sanctuary", "dung_gate_portal", "zion_gate_portal"]
    temple_ids = ["dome_of_the_rock_south", "dome_of_the_chain", "al_aqsa_mosque_porch", "western_wall_plaza", "holy_sepulchre_parvis", "muristan_redeemer", "hurva_synagogue_plaza", "st_james_cathedral"]
    tour_ids = ["tower_of_david", "omar_square_north", "david_street_souq", "triple_souq_cardo", "chain_street_turba", "southern_roman_cardo", "ecce_homo_via_dolorosa"]

    make_contact_sheet([out_dir / f"{i}.jpg" for i in gate_ids], out_dir / "contact_gates.jpg", cols=3)
    make_contact_sheet([out_dir / f"{i}.jpg" for i in temple_ids], out_dir / "contact_temples.jpg", cols=3)
    make_contact_sheet([out_dir / f"{i}.jpg" for i in tour_ids], out_dir / "contact_touristic.jpg", cols=3)
    print(f"[OK] Built POI pairs and contact sheets in {out_dir}")


if __name__ == "__main__":
    main()
