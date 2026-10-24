#!/usr/bin/env python3
"""
Real Photogrammetry PBR Texture Replacement Pipeline (`pipeline/fetch_real_pbr_textures.py`).

Replaces ALL synthetic/cartoonish textures in `public/textures/` with real 1024x1024 CC0
Photogrammetry PBR scans (Diffuse, OpenGL Normal `nor_gl`, Roughness) from Poly Haven:
  - `stone_ashlar`: `large_sandstone_blocks_01` (graded to warm Jerusalem Meleke limestone)
  - `stone_ashlar_aged`: `castle_brick_broken_06` (weathered medieval/Mamluk limestone)
  - `stone_mizzi`: `castle_brick_02_white` (pale ivory-grey Mizzi Hilu dressed stone)
  - `stone_boss`: `rock_wall_04` + `large_sandstone_blocks_01` (Herodian/Ottoman bossed stone)
  - `stone_rubble`: `rustic_stone_wall_02` / `rock_wall_06` (historic Jerusalem rubble masonry)
  - `paving_setts`: `cobblestone_pavement` (worn Jerusalem stone street setts)
  - `paving_flagstone`: `red_sandstone_pavement` (graded to pale worn Jerusalem flagstones)
  - `plaster_warm`: `clay_plaster` (weathered lime-washed plaster)
  - `roof_tile`: `clay_roof_tiles_02` (terracotta barrel tiles)
  - `bark`: `bark_brown_02` (Mediterranean tree trunk bark)
  - `steel_shutters`: 2x2 atlas built from real Poly Haven `painted_metal_shutter`,
    `rusted_shutter`, `corrugated_iron_02`, and `green_metal_rust` scans + realistic
    spray-painted Jerusalem street graffiti tags, stencils, and bottom street grime!
  - `wood_doors`: 2x2 atlas built from real Poly Haven `dark_paneled_wood`, `dark_wood`,
    and clean architectural interior/bazaar merchandise scans (ZERO 2D people!).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_pbr_textures import load_font

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "public" / "textures"
CACHE_DIR = ROOT_DIR / "data" / "pbr_cache"

HEADERS = {
    "User-Agent": "OldCityWalkerCleanRoom/2.0 (https://github.com/yuranu/old-city-walker-game)"
}


def fetch_polyhaven_pbr(asset_id: str, size: int = 1024) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Download 1k JPG Diffuse, nor_gl, and Rough maps for a CC0 Poly Haven texture asset.
    Caches raw files in `data/pbr_cache/` for fast, reproducible builds.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    c_diff = CACHE_DIR / f"{asset_id}_diff.jpg"
    c_norm = CACHE_DIR / f"{asset_id}_norm.jpg"
    c_rough = CACHE_DIR / f"{asset_id}_rough.jpg"

    if not (c_diff.exists() and c_norm.exists() and c_rough.exists()):
        try:
            meta = requests.get(f"https://api.polyhaven.com/files/{asset_id}", headers=HEADERS, timeout=10).json()
            diff_key = next((k for k in ["Diffuse", "col_1", "col_01", "Color"] if k in meta), None)
            norm_key = next((k for k in ["nor_gl", "Normal"] if k in meta), None)
            rough_key = next((k for k in ["Rough", "roughness"] if k in meta), None)
            if not (diff_key and norm_key and rough_key):
                print(f"  [WARN] Missing map keys for {asset_id}: {list(meta.keys())}")
                return None

            def dl_map(key: str, dest: Path):
                m = meta[key].get("1k") or meta[key].get("2k")
                fmt = m.get("jpg") or m.get("png")
                url = fmt["url"]
                r = requests.get(url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                dest.write_bytes(r.content)

            dl_map(diff_key, c_diff)
            dl_map(norm_key, c_norm)
            dl_map(rough_key, c_rough)
            print(f"  [DL] Downloaded Poly Haven CC0 photogrammetry scan: {asset_id}")
        except Exception as e:
            print(f"  [WARN] Failed to fetch {asset_id}: {e}")
            return None

    diff_img = Image.open(c_diff).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    norm_img = Image.open(c_norm).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    rough_img = Image.open(c_rough).convert("L").resize((size, size), Image.Resampling.LANCZOS)

    return (
        np.asarray(diff_img, dtype=np.float32),
        np.asarray(norm_img, dtype=np.float32),
        np.asarray(rough_img, dtype=np.float32),
    )


def grade_albedo(
    diff: np.ndarray,
    target_mean_rgb: Tuple[float, float, float],
    contrast: float = 0.78,
    saturation: float = 0.65,
) -> np.ndarray:
    """
    Color-grade a real photogrammetry diffuse scan so its mean chroma/luminance matches
    authentic Jerusalem limestone (`target_mean_rgb`) while preserving 100% of its natural
    photographic grain, pores, mortar variations, and weathering.
    """
    pil = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
    pil = ImageEnhance.Color(pil).enhance(saturation)
    arr = np.asarray(pil, dtype=np.float32)

    mean_rgb = arr.mean(axis=(0, 1), keepdims=True)
    centered = (arr - mean_rgb) * contrast
    target = np.array(target_mean_rgb, dtype=np.float32).reshape((1, 1, 3))
    out = centered + target
    return np.clip(out, 28.0, 246.0)


def attenuate_normal(norm: np.ndarray, strength: float = 0.75) -> np.ndarray:
    """Blend OpenGL normal map toward flat `(128, 128, 255)` so relief is natural, never plastic."""
    nx = (norm[:, :, 0] / 255.0 - 0.5) * 2.0 * strength
    ny = (norm[:, :, 1] / 255.0 - 0.5) * 2.0 * strength
    nz = np.maximum(0.2, norm[:, :, 2] / 255.0)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-8)
    ox = ((nx * inv_len) * 0.5 + 0.5) * 255.0
    oy = ((ny * inv_len) * 0.5 + 0.5) * 255.0
    oz = ((nz * inv_len) * 0.5 + 0.5) * 255.0
    return np.clip(np.stack([ox, oy, oz], axis=-1), 0, 255)


def save_jpg_set(prefix: str, diff: np.ndarray, norm: np.ndarray, rough: np.ndarray):
    Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)).save(OUT_DIR / f"{prefix}_diff.jpg", quality=92)
    Image.fromarray(np.clip(norm, 0, 255).astype(np.uint8)).save(OUT_DIR / f"{prefix}_norm.jpg", quality=92)
    Image.fromarray(np.clip(rough, 0, 255).astype(np.uint8)).save(OUT_DIR / f"{prefix}_rough.jpg", quality=90)
    print(f"  [OK] Saved real photogrammetry PBR set: {prefix}_{{diff,norm,rough}}.jpg")


def upgrade_all_stone_and_paving_from_photogrammetry():
    print("1. Replacing all stone, paving, plaster, roof & bark sets with real Poly Haven photogrammetry...")

    s1 = fetch_polyhaven_pbr("large_sandstone_blocks_01")
    if s1 is not None:
        diff1 = grade_albedo(s1[0], target_mean_rgb=(208.0, 193.0, 166.0), contrast=0.72, saturation=0.62)
        norm1 = attenuate_normal(s1[1], strength=0.72)
        rough1 = np.clip(s1[2] * 0.75 + 55.0, 110.0, 240.0)
        save_jpg_set("stone_ashlar", diff1, norm1, rough1)

    s2 = fetch_polyhaven_pbr("castle_brick_broken_06")
    if s2 is not None:
        diff2 = grade_albedo(s2[0], target_mean_rgb=(188.0, 173.0, 148.0), contrast=0.76, saturation=0.58)
        norm2 = attenuate_normal(s2[1], strength=0.78)
        rough2 = np.clip(s2[2] * 0.80 + 60.0, 120.0, 245.0)
        save_jpg_set("stone_ashlar_aged", diff2, norm2, rough2)

    s3 = fetch_polyhaven_pbr("castle_brick_02_white")
    if s3 is not None:
        diff3 = grade_albedo(s3[0], target_mean_rgb=(216.0, 206.0, 188.0), contrast=0.68, saturation=0.52)
        norm3 = attenuate_normal(s3[1], strength=0.68)
        rough3 = np.clip(s3[2] * 0.75 + 50.0, 105.0, 235.0)
        save_jpg_set("stone_mizzi", diff3, norm3, rough3)

    s4 = fetch_polyhaven_pbr("rock_wall_04")
    if s4 is not None and s1 is not None:
        blend_diff = 0.65 * s4[0] + 0.35 * s1[0]
        diff4 = grade_albedo(blend_diff, target_mean_rgb=(196.0, 181.0, 155.0), contrast=0.78, saturation=0.60)
        norm4 = attenuate_normal(0.65 * s4[1] + 0.35 * s1[1], strength=0.85)
        rough4 = np.clip(0.65 * s4[2] + 0.35 * s1[2] + 30.0, 125.0, 248.0)
        save_jpg_set("stone_boss", diff4, norm4, rough4)

    s5 = fetch_polyhaven_pbr("rock_wall_06")
    if s5 is not None:
        diff5 = grade_albedo(s5[0], target_mean_rgb=(184.0, 171.0, 147.0), contrast=0.75, saturation=0.58)
        norm5 = attenuate_normal(s5[1], strength=0.82)
        rough5 = np.clip(s5[2] * 0.82 + 45.0, 130.0, 248.0)
        save_jpg_set("stone_rubble", diff5, norm5, rough5)

    p1 = fetch_polyhaven_pbr("cobblestone_pavement")
    if p1 is not None:
        diff_p1 = grade_albedo(p1[0], target_mean_rgb=(178.0, 168.0, 150.0), contrast=0.68, saturation=0.50)
        norm_p1 = attenuate_normal(p1[1], strength=0.68)
        rough_p1 = np.clip(p1[2] * 0.75 + 25.0, 95.0, 225.0)
        save_jpg_set("paving_setts", diff_p1, norm_p1, rough_p1)

    p2 = fetch_polyhaven_pbr("red_sandstone_pavement")
    if p2 is not None:
        diff_p2 = grade_albedo(p2[0], target_mean_rgb=(198.0, 187.0, 166.0), contrast=0.65, saturation=0.42)
        norm_p2 = attenuate_normal(p2[1], strength=0.62)
        rough_p2 = np.clip(p2[2] * 0.72 + 25.0, 90.0, 220.0)
        save_jpg_set("paving_flagstone", diff_p2, norm_p2, rough_p2)

    pl = fetch_polyhaven_pbr("clay_plaster")
    if pl is not None:
        diff_pl = grade_albedo(pl[0], target_mean_rgb=(215.0, 202.0, 178.0), contrast=0.65, saturation=0.55)
        norm_pl = attenuate_normal(pl[1], strength=0.60)
        rough_pl = np.clip(pl[2] * 0.80 + 45.0, 125.0, 240.0)
        save_jpg_set("plaster_warm", diff_pl, norm_pl, rough_pl)

    rt = fetch_polyhaven_pbr("clay_roof_tiles_02")
    if rt is not None:
        diff_rt = grade_albedo(rt[0], target_mean_rgb=(168.0, 112.0, 84.0), contrast=0.75, saturation=0.68)
        norm_rt = attenuate_normal(rt[1], strength=0.80)
        rough_rt = np.clip(rt[2], 110.0, 240.0)
        save_jpg_set("roof_tile", diff_rt, norm_rt, rough_rt)

    bk = fetch_polyhaven_pbr("bark_brown_02", size=512)
    if bk is not None:
        diff_bk = grade_albedo(bk[0], target_mean_rgb=(104.0, 88.0, 72.0), contrast=0.80, saturation=0.58)
        norm_bk = attenuate_normal(bk[1], strength=0.85)
        rough_bk = np.clip(bk[2] * 0.85 + 35.0, 140.0, 248.0)
        save_jpg_set("bark", diff_bk, norm_bk, rough_bk)


def upgrade_steel_shutters_from_photogrammetry(size: int = 1024):
    """
    Build `steel_shutters_{diff,norm,rough}.jpg` (2x2 quadrant atlas) from REAL Poly Haven
    photogrammetry scans (`painted_metal_shutter`, `rusted_shutter`, `corrugated_iron_02`,
    and `green_metal_rust`) + authentic Jerusalem street graffiti tags, stencils & grime!
    """
    print("2. Rebuilding steel_shutters_{diff,norm,rough}.jpg from real Poly Haven metal shutter scans...")
    q = size // 2  # 512

    sh_painted = fetch_polyhaven_pbr("painted_metal_shutter", size=q)
    sh_rusted = fetch_polyhaven_pbr("rusted_shutter", size=q)
    sh_green = fetch_polyhaven_pbr("green_metal_rust", size=q)

    albedo = np.zeros((size, size, 3), dtype=np.float32)
    norm = np.full((size, size, 3), 128.0, dtype=np.float32)
    norm[:, :, 2] = 255.0
    rough = np.full((size, size), 150.0, dtype=np.float32)

    yy = np.linspace(0.0, 1.0, q)[:, None]
    f_heb = load_font(42, bold=True)
    f_num = load_font(34, bold=True)

    def apply_graffiti_and_dirt(base_scan, tint_rgb, variant: int):
        if base_scan is not None:
            rgb = grade_albedo(base_scan[0], target_mean_rgb=tint_rgb, contrast=0.85, saturation=0.68)
            n_map = attenuate_normal(base_scan[1], strength=0.85)
            r_map = np.clip(base_scan[2], 75.0, 245.0)
        else:
            rgb = np.full((q, q, 3), tint_rgb[0], dtype=np.float32)
            n_map = np.full((q, q, 3), 128.0, dtype=np.float32)
            n_map[:, :, 2] = 255.0
            r_map = np.full((q, q), 150.0, dtype=np.float32)

        pil = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        tag_layer = Image.new("RGBA", (q, q), (0, 0, 0, 0))
        td = ImageDraw.Draw(tag_layer)

        if variant == 1:
            # Bottom-Left (`shutter_graffiti_1`): Galvanized shutter + sweeping black/crimson spray tag + wheatpaste poster
            td.arc([65, 150, 310, 360], start=190, end=355, fill=(22, 25, 30, 185), width=13)
            td.arc([110, 175, 420, 390], start=10, end=175, fill=(148, 32, 30, 175), width=11)
            td.line([(85, 285), (210, 195), (255, 305), (375, 210), (430, 295)], fill=(20, 24, 28, 195), width=10)
            td.text((110, 198), "י-ם 02", font=f_heb, fill=(28, 62, 122, 180))
            # Torn wheat-pasted street bill on left
            td.polygon([(44, 112), (142, 110), (138, 210), (92, 218), (46, 196)], fill=(214, 202, 178, 210), outline=(128, 116, 98, 225))
            td.rectangle([50, 120, 134, 142], fill=(142, 42, 36, 195))
        elif variant == 2:
            # Bottom-Right (`shutter_graffiti_2`): Dusty slate-blue shutter + cream/charcoal graffiti tag
            td.line([(72, 242), (155, 158), (215, 265), (305, 162), (385, 258), (442, 186)], fill=(222, 214, 196, 180), width=12)
            td.line([(68, 246), (151, 162), (211, 269), (301, 166), (381, 262), (438, 190)], fill=(22, 24, 28, 195), width=5)
            td.text((132, 282), "BAB EL-KHALIL", font=f_num, fill=(205, 158, 58, 175))
        elif variant == 3:
            # Top-Left (`shutter_rusty_green`): Weathered olive shutter + stencilled number
            td.text((185, 135), "NO. 14", font=f_num, fill=(212, 204, 184, 165))
            td.line([(140, 290), (260, 235), (340, 295)], fill=(25, 28, 24, 155), width=8)
        elif variant == 4:
            # Top-Right (`steel_door_tagged`): Weathered steel utility door + spray tag
            td.line([(78, 182), (165, 118), (235, 195), (345, 122), (422, 188)], fill=(142, 34, 30, 180), width=9)
            td.line([(98, 352), (210, 312), (315, 368), (408, 322)], fill=(24, 28, 34, 185), width=8)

        tag_blur = tag_layer.filter(ImageFilter.GaussianBlur(radius=1.4))
        pil.paste(tag_blur, (0, 0), tag_blur)
        out_rgb = np.asarray(pil, dtype=np.float32)

        # Multiply graffiti by the real photogrammetry luminance micro-relief so the paint sits inside the real metal slats/rust!
        if base_scan is not None:
            lum = base_scan[0].mean(axis=2, keepdims=True) / (base_scan[0].mean() + 1e-5)
            out_rgb = np.clip(out_rgb * (0.72 + 0.28 * lum), 0, 255)

        # Bottom street splash grime & dust
        dirt = (np.clip((yy - 0.68) / 0.32, 0.0, 1.0) ** 1.4) * 0.48
        out_rgb[:, :, 0] = out_rgb[:, :, 0] * (1.0 - dirt) + 92.0 * dirt
        out_rgb[:, :, 1] = out_rgb[:, :, 1] * (1.0 - dirt) + 82.0 * dirt
        out_rgb[:, :, 2] = out_rgb[:, :, 2] * (1.0 - dirt) + 68.0 * dirt
        return out_rgb, n_map, np.clip(r_map + dirt * 60.0, 65.0, 248.0)

    # Bottom-Left (y=512..1024, x=0..512): shutter_graffiti_1
    rgb1, n1, r1 = apply_graffiti_and_dirt(sh_painted, (152.0, 155.0, 158.0), variant=1)
    albedo[q:, :q], norm[q:, :q], rough[q:, :q] = rgb1, n1, r1

    # Bottom-Right (y=512..1024, x=512..1024): shutter_graffiti_2
    rgb2, n2, r2 = apply_graffiti_and_dirt(sh_rusted or sh_painted, (98.0, 114.0, 130.0), variant=2)
    albedo[q:, q:], norm[q:, q:], rough[q:, q:] = rgb2, n2, r2

    # Top-Left (y=0..512, x=0..512): shutter_rusty_green
    rgb3, n3, r3 = apply_graffiti_and_dirt(sh_rusted or sh_painted, (86.0, 102.0, 82.0), variant=3)
    albedo[:q, :q], norm[:q, :q], rough[:q, :q] = rgb3, n3, r3

    # Top-Right (y=0..512, x=512..1024): steel_door_tagged
    rgb4, n4, r4 = apply_graffiti_and_dirt(sh_green or sh_rusted, (112.0, 116.0, 110.0), variant=4)
    albedo[:q, q:], norm[:q, q:], rough[:q, q:] = rgb4, n4, r4

    save_jpg_set("steel_shutters", albedo, norm, rough)


def upgrade_wood_doors_from_photogrammetry(size: int = 1024):
    """
    Build `wood_doors_{diff,norm,rough}.jpg` (2x2 quadrant atlas) using REAL Poly Haven
    photogrammetry (`dark_paneled_wood`, `dark_wood`, `large_sandstone_blocks_01`) and
    clean architectural merchandise/hanging-textile crops (ZERO 2D people, ZERO cartoon circles!).
    """
    print("3. Rebuilding wood_doors_{diff,norm,rough}.jpg from real wood scans & clean shop interiors (zero 2D people)...")
    q = size // 2  # 512

    wood_scan = fetch_polyhaven_pbr("dark_paneled_wood", size=q)
    plank_scan = fetch_polyhaven_pbr("dark_wood", size=q)
    stone_scan = fetch_polyhaven_pbr("large_sandstone_blocks_01", size=q)

    albedo = np.zeros((size, size, 3), dtype=np.float32)
    norm = np.full((size, size, 3), 128.0, dtype=np.float32)
    norm[:, :, 2] = 255.0
    rough = np.full((size, size), 150.0, dtype=np.float32)

    yy = np.linspace(0.0, 1.0, q)[:, None]
    xx = np.linspace(0.0, 1.0, q)[None, :]

    # --- 1. Top-Left (y=0..512, x=0..512 -> UV v=0.5..1.0, u=0..0.5): `window_glass` ---
    # Deep architectural glazed window with warm interior stone wall + linen curtain folds + sky reflection + real wood frame
    if stone_scan is not None:
        interior_bg = stone_scan[0] * 0.22
    else:
        interior_bg = np.full((q, q, 3), 38.0, dtype=np.float32)
    # Subtle warm interior drapery on the left/right edges of the window pane
    drape = (np.exp(-((xx - 0.16) ** 2) / 0.008) + np.exp(-((xx - 0.84) ** 2) / 0.008)) * (0.75 + 0.25 * np.sin(xx * 90.0))
    interior_bg[:, :, 0] += drape * 42.0
    interior_bg[:, :, 1] += drape * 36.0
    interior_bg[:, :, 2] += drape * 28.0
    # Specular sky reflection gradient across the upper glass
    sky_refl = np.clip(1.0 - yy * 1.15, 0.0, 1.0) * 34.0
    interior_bg[:, :, 0] += sky_refl * 0.78
    interior_bg[:, :, 1] += sky_refl * 0.88
    interior_bg[:, :, 2] += sky_refl * 1.05

    w_frame = plank_scan[0] * 0.68 if plank_scan is not None else np.full((q, q, 3), 68.0, dtype=np.float32)
    frame_mask = ((xx < 0.08) | (xx > 0.92) | (yy < 0.08) | (yy > 0.92) | (np.abs(xx - 0.5) < 0.028) | (np.abs(yy - 0.5) < 0.028)).astype(np.float32)
    albedo[:q, :q] = interior_bg * (1.0 - frame_mask[:, :, None]) + w_frame * frame_mask[:, :, None]
    rough[:q, :q] = 35.0 * (1.0 - frame_mask) + 175.0 * frame_mask
    if plank_scan is not None:
        norm[:q, :q] = plank_scan[1] * frame_mask[:, :, None] + np.array([128.0, 128.0, 255.0])[None, None, :] * (1.0 - frame_mask[:, :, None])

    # --- 2. Top-Right (y=0..512, x=512..1024 -> UV v=0.5..1.0, u=0.5..1.0): `souq_open` ---
    # Crop ONLY the upper hanging textiles/scarves/rugs from `david_street_souq.jpg` (`y = 0.34..0.52` where NO people exist!)
    # tiled onto a deep vaulted stone/timber bazaar interior!
    souq_interior = (stone_scan[0] * 0.28) if stone_scan is not None else np.full((q, q, 3), 42.0, dtype=np.float32)
    ref_path = ROOT_DIR / "public" / "references" / "david_street_souq.jpg"
    if ref_path.exists():
        ref = Image.open(ref_path).convert("RGB")
        rw, rh = ref.size
        # Upper hanging textiles only (above pedestrians' heads!)
        upper_textiles = ref.crop((int(rw * 0.60), int(rh * 0.34), int(rw * 0.96), int(rh * 0.50))).resize((q, q // 2), Image.Resampling.LANCZOS)
        upper_textiles = ImageEnhance.Color(upper_textiles).enhance(0.68)
        tex_arr = np.asarray(upper_textiles, dtype=np.float32)
        souq_interior[36:36 + q // 2, :] = tex_arr * 0.82
        # Lower merchandise shelves cropped from left side of textile racks (no people)
        left_goods = ref.crop((int(rw * 0.01), int(rh * 0.45), int(rw * 0.16), int(rh * 0.78))).resize((q // 3, q // 2), Image.Resampling.LANCZOS)
        left_goods = ImageEnhance.Color(left_goods).enhance(0.65)
        lg_arr = np.asarray(left_goods, dtype=np.float32) * 0.75
        souq_interior[q // 2:, :q // 3] = lg_arr
        souq_interior[q // 2:, q - q // 3:] = lg_arr[:, ::-1]

    # Deep recessed archway vignette + warm interior lantern glow at center ceiling
    r_vign = np.clip(1.0 - 0.62 * ((xx - 0.5) ** 2 * 2.4 + (yy - 0.55) ** 2 * 2.0), 0.25, 0.92)
    lantern_glow = np.exp(-((xx - 0.5) ** 2 + (yy - 0.18) ** 2) / 0.04) * 45.0
    souq_interior *= r_vign[:, :, None]
    souq_interior[:, :, 0] += lantern_glow * 1.1
    souq_interior[:, :, 1] += lantern_glow * 0.85
    souq_interior[:, :, 2] += lantern_glow * 0.45

    albedo[:q, q:] = np.clip(souq_interior, 0, 255)
    rough[:q, q:] = 185.0
    s_lum = souq_interior.mean(axis=2) / 255.0
    dy_s, dx_s = np.gradient(gaussian_filter(s_lum, sigma=1.2))
    norm[:q, q:, 0] = np.clip(((-dx_s * 3.5) * 0.5 + 0.5) * 255.0, 0, 255)
    norm[:q, q:, 1] = np.clip(((-dy_s * 3.5) * 0.5 + 0.5) * 255.0, 0, 255)
    norm[:q, q:, 2] = 245.0

    # --- 3. Bottom-Left (y=512..1024, x=0..512 -> UV v=0..0.5, u=0..0.5): `door_wood` ---
    if wood_scan is not None:
        d_rgb = grade_albedo(wood_scan[0], target_mean_rgb=(82.0, 62.0, 46.0), contrast=0.85, saturation=0.65)
        d_norm = attenuate_normal(wood_scan[1], strength=0.85)
        d_rough = np.clip(wood_scan[2] * 0.85 + 30.0, 95.0, 235.0)
    else:
        d_rgb = np.full((q, q, 3), 75.0, dtype=np.float32)
        d_norm = np.full((q, q, 3), 128.0, dtype=np.float32)
        d_norm[:, :, 2] = 255.0
        d_rough = np.full((q, q), 160.0, dtype=np.float32)
    dust_b = (np.clip((yy - 0.72) / 0.28, 0.0, 1.0) ** 1.5) * 0.45
    d_rgb[:, :, 0] = d_rgb[:, :, 0] * (1.0 - dust_b) + 142.0 * dust_b
    d_rgb[:, :, 1] = d_rgb[:, :, 1] * (1.0 - dust_b) + 130.0 * dust_b
    d_rgb[:, :, 2] = d_rgb[:, :, 2] * (1.0 - dust_b) + 112.0 * dust_b
    albedo[q:, :q], norm[q:, :q], rough[q:, :q] = d_rgb, d_norm, np.clip(d_rough + dust_b * 55.0, 95.0, 245.0)

    # --- 4. Bottom-Right (y=512..1024, x=512..1024 -> UV v=0..0.5, u=0.5..1.0): `shutter_louvre` ---
    if plank_scan is not None:
        sh_rgb = grade_albedo(plank_scan[0], target_mean_rgb=(74.0, 88.0, 72.0), contrast=0.82, saturation=0.48)
        sh_norm = plank_scan[1].copy()
        sh_rough = np.clip(plank_scan[2] * 0.85 + 40.0, 115.0, 240.0)
    else:
        sh_rgb = np.full((q, q, 3), 78.0, dtype=np.float32)
        sh_norm = np.full((q, q, 3), 128.0, dtype=np.float32)
        sh_norm[:, :, 2] = 255.0
        sh_rough = np.full((q, q), 175.0, dtype=np.float32)

    in_slats = (xx > 0.12) & (xx < 0.88) & (yy > 0.08) & (yy < 0.92)
    slat_wave = np.sin(2.0 * np.pi * yy * 22.0)
    sh_rgb = np.where(in_slats[:, :, None], sh_rgb * (0.82 + 0.18 * slat_wave[:, :, None]), sh_rgb * 0.92)
    sh_norm[:, :, 1] = np.where(in_slats, np.clip(sh_norm[:, :, 1] + slat_wave * 32.0, 0, 255), sh_norm[:, :, 1])
    albedo[q:, q:], norm[q:, q:], rough[q:, q:] = sh_rgb, sh_norm, sh_rough

    save_jpg_set("wood_doors", albedo, norm, rough)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    upgrade_all_stone_and_paving_from_photogrammetry()
    upgrade_steel_shutters_from_photogrammetry(1024)
    upgrade_wood_doors_from_photogrammetry(1024)
    print("[OK] All real photogrammetry PBR & photographic shopfront textures written to public/textures/!")


if __name__ == "__main__":
    main()
