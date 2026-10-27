#!/usr/bin/env python3
"""
scripts/make_roam_sheets.py
Generates side-by-side REFERENCE PHOTO | GAME SCREENSHOT pairs (`pair_<refId>.jpg`)
and labeled zone contact sheets (`contact_zone1_west_and_gate.jpg`, etc.) for AI Visual Agents.
"""

import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: make_roam_sheets.py <out_dir>")
        sys.exit(1)

    out_dir = Path(sys.argv[1]).resolve()
    root_dir = Path(__file__).resolve().parent.parent
    ref_dir = root_dir / "public" / "references"
    manifest_path = out_dir / "roam_manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    # 1. Build side-by-side REAL | MODEL pairs where refId is present
    for v in manifest:
        ref_id = v.get("refId")
        if not ref_id:
            continue
        ref_path = ref_dir / f"{ref_id}.jpg"
        if not ref_path.exists():
            ref_path = root_dir / "data" / "references" / f"{ref_id}.jpg"
        mod_path = out_dir / v["file"]
        if not ref_path.exists() or not mod_path.exists():
            continue
        ref_im = Image.open(ref_path).convert("RGB").resize((960, 600), Image.Resampling.LANCZOS)
        mod_im = Image.open(mod_path).convert("RGB").resize((960, 600), Image.Resampling.LANCZOS)
        pair = Image.new("RGB", (1920, 648), (18, 20, 26))
        pair.paste(ref_im, (0, 48))
        pair.paste(mod_im, (960, 48))
        dr = ImageDraw.Draw(pair)
        dr.text((16, 14), f"REFERENCE PHOTO: {ref_id} ({v.get('name', '')})", fill=(130, 220, 255), font=font)
        dr.text(
            (976, 14),
            f"GAME SCREENSHOT: ({v.get('x')}m, {v.get('y')}m, hdg={v.get('heading')}, pitch={v.get('pitch')})",
            fill=(255, 220, 130),
            font=font,
        )
        p_name = f"pair_{ref_id}.jpg"
        pair.save(out_dir / p_name, quality=92)

    # 2. Build contact sheets
    def make_sheet(items: list, sheet_name: str) -> None:
        if not items:
            return
        cols = 2
        rows = (len(items) + cols - 1) // cols
        tw, th, pad, hdr = 640, 400, 12, 36
        sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + hdr + pad) + pad), (15, 17, 22))
        dr = ImageDraw.Draw(sheet)
        for idx, v in enumerate(items):
            p = out_dir / v["file"]
            if not p.exists():
                continue
            c = idx % cols
            r = idx // cols
            x0 = pad + c * (tw + pad)
            y0 = pad + r * (th + hdr + pad)
            im = Image.open(p).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
            sheet.paste(im, (x0, y0 + hdr))
            label = f"{v['id']} | ({v.get('x')}, {v.get('y')}, hdg={v.get('heading')}, p={v.get('pitch')}) | {v.get('name', '')}"
            dr.text((x0 + 6, y0 + 8), label[:82], fill=(240, 240, 245), font=font_sm)
        sheet.save(out_dir / sheet_name, quality=90)

    z1 = [v for v in manifest if str(v.get("id", "")).startswith("z1_")]
    z2 = [v for v in manifest if str(v.get("id", "")).startswith("z2_")]
    z3 = [v for v in manifest if str(v.get("id", "")).startswith("z3_")]
    if z1 and not z2 and not z3 and not any(v.get("id", "").startswith("z1_01_") for v in z1):
        make_sheet(manifest, "contact_custom.jpg")
    elif z1 or z2 or z3:
        make_sheet(z1, "contact_zone1_west_and_gate.jpg")
        make_sheet(z2, "contact_zone2_citadel_and_omar_sq.jpg")
        make_sheet(z3, "contact_zone3_latin_and_david_souq.jpg")
    else:
        make_sheet(manifest, "contact_custom.jpg")


if __name__ == "__main__":
    main()
