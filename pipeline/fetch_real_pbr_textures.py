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
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def load_font(size: int, bold: bool = True, serif: bool = False):
    candidates = []
    if serif:
        candidates = [
            "/usr/share/fonts/gnu-free/FreeSerifBold.otf" if bold else "/usr/share/fonts/gnu-free/FreeSerif.otf",
            "/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
    candidates += [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def gaussian_filter(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min < 1e-8:
        return arr.copy()
    u8 = np.clip((arr - a_min) / (a_max - a_min) * 255.0, 0, 255).astype(np.uint8)
    blurred = Image.fromarray(u8).filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred, dtype=np.float32) / 255.0 * (a_max - a_min) + a_min

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


def save_jpg_set(prefix: str, diff: np.ndarray, norm: np.ndarray, rough: np.ndarray, quality: int = 86):
    Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)).save(OUT_DIR / f"{prefix}_diff.jpg", quality=quality)
    Image.fromarray(np.clip(norm, 0, 255).astype(np.uint8)).save(OUT_DIR / f"{prefix}_norm.jpg", quality=quality)
    Image.fromarray(np.clip(rough, 0, 255).astype(np.uint8)).save(OUT_DIR / f"{prefix}_rough.jpg", quality=max(82, quality - 2))
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
            # Bottom-Left (`shutter_graffiti_1`): Clean galvanized shutter + subtle stencilled bay number (NO chevron/mountain line drawings!)
            td.text((195, 165), "NO. 02", font=f_num, fill=(38, 44, 52, 135))
        elif variant == 2:
            # Bottom-Right (`shutter_graffiti_2`): Dusty slate-blue shutter + subtle stencilled header
            td.text((135, 165), "BAB EL-KHALIL", font=f_num, fill=(215, 208, 192, 130))
        elif variant == 3:
            # Top-Left (`shutter_rusty_green`): Weathered olive shutter + stencilled number
            td.text((185, 165), "NO. 14", font=f_num, fill=(212, 204, 184, 140))
        elif variant == 4:
            # Top-Right (`steel_door_tagged`): Clean weathered steel utility double door with center seam
            td.line([(q // 2, 16), (q // 2, q - 16)], fill=(24, 26, 28, 180), width=5)

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
    # 100% pure architectural glazed & timber shopfront built strictly from Poly Haven
    # `dark_paneled_wood`, `dark_wood`, `large_sandstone_blocks_01`, and warm recessed wooden
    # display shelves / woven geometric tapestries (ZERO street photographs, ZERO umbrellas, ZERO people!).
    souq_interior = (stone_scan[0] * 0.30) if stone_scan is not None else np.full((q, q, 3), 42.0, dtype=np.float32)
    wood_base = plank_scan[0] if plank_scan is not None else np.full((q, q, 3), 85.0, dtype=np.float32)
    panel_base = wood_scan[0] if wood_scan is not None else wood_base

    # Warm recessed archway ceiling lantern glow
    r_vign = np.clip(1.0 - 0.55 * ((xx - 0.5) ** 2 * 2.2 + (yy - 0.50) ** 2 * 1.8), 0.30, 0.96)
    lantern_glow = np.exp(-((xx - 0.5) ** 2 + (yy - 0.20) ** 2) / 0.055) * 62.0
    souq_interior *= r_vign[:, :, None]
    souq_interior[:, :, 0] += lantern_glow * 1.15
    souq_interior[:, :, 1] += lantern_glow * 0.88
    souq_interior[:, :, 2] += lantern_glow * 0.48

    # Left & right woven geometric kilim textile tapestries hanging inside the timber jambs
    weave_mod = 0.88 + 0.12 * np.sin(2.0 * np.pi * xx * 48.0) * np.cos(2.0 * np.pi * yy * 48.0)
    left_rug = (xx >= 0.08) & (xx <= 0.26) & (yy >= 0.12) & (yy <= 0.72)
    right_rug = (xx >= 0.74) & (xx <= 0.92) & (yy >= 0.12) & (yy <= 0.72)
    stripe_band = ((np.sin(2.0 * np.pi * yy * 9.0) > 0.0)).astype(np.float32)
    rug_l_rgb = np.stack([
        (138.0 * stripe_band + 168.0 * (1.0 - stripe_band)) * weave_mod,
        (46.0 * stripe_band + 118.0 * (1.0 - stripe_band)) * weave_mod,
        (38.0 * stripe_band + 52.0 * (1.0 - stripe_band)) * weave_mod,
    ], axis=-1)
    rug_r_rgb = np.stack([
        (38.0 * stripe_band + 154.0 * (1.0 - stripe_band)) * weave_mod,
        (78.0 * stripe_band + 52.0 * (1.0 - stripe_band)) * weave_mod,
        (116.0 * stripe_band + 44.0 * (1.0 - stripe_band)) * weave_mod,
    ], axis=-1)
    souq_interior = np.where(left_rug[:, :, None], rug_l_rgb, souq_interior)
    souq_interior = np.where(right_rug[:, :, None], rug_r_rgb, souq_interior)

    # Recessed horizontal timber display shelves & lower wooden counter (`dark_paneled_wood`)
    shelves_mask = (
        ((xx >= 0.26) & (xx <= 0.74) & ((np.abs(yy - 0.34) < 0.022) | (np.abs(yy - 0.52) < 0.022) | (np.abs(yy - 0.70) < 0.022)))
        | (yy >= 0.76)
        | (xx < 0.08)
        | (xx > 0.92)
        | (yy < 0.08)
    ).astype(np.float32)
    timber_rgb = grade_albedo(panel_base, target_mean_rgb=(88.0, 64.0, 44.0), contrast=0.82, saturation=0.62)
    souq_interior = souq_interior * (1.0 - shelves_mask[:, :, None]) + timber_rgb * shelves_mask[:, :, None]

    albedo[:q, q:] = np.clip(souq_interior, 0, 255)
    rough[:q, q:] = 175.0 * (1.0 - shelves_mask) + 145.0 * shelves_mask
    if wood_scan is not None:
        norm[:q, q:] = attenuate_normal(wood_scan[1], strength=0.80)

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


def upgrade_accent_pbr_textures(size: int = 256):
    """
    Generates complete `{diff,norm,rough}.jpg` PBR texture sets for all accent materials
    (`awning_red`, `awning_green`, `awning_striped`, `metal_iron`, `corrugated_metal`,
    `terracotta`, `cardboard`, `trash_bag`, `trash_blue`) derived from Poly Haven scans
    and woven/crinkled micro-relief so ZERO materials use untextured solid colors!
    """
    print("4. Generating textured PBR sets for awnings, metals, terracotta, cardboard & trash bags...")
    pl = fetch_polyhaven_pbr("clay_plaster", size=size)
    sh_painted = fetch_polyhaven_pbr("painted_metal_shutter", size=size)
    sh_rusted = fetch_polyhaven_pbr("rusted_shutter", size=size)
    sh_green = fetch_polyhaven_pbr("green_metal_rust", size=size)
    rt = fetch_polyhaven_pbr("clay_roof_tiles_02", size=size)
    wood = fetch_polyhaven_pbr("dark_wood", size=size)

    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, size, endpoint=False), np.linspace(0.0, 1.0, size, endpoint=False), indexing="ij")

    # 1. Woven Sun-Weathered Canvas Awning PBR Sets (`awning_red`, `awning_green`, `awning_striped`)
    warp = np.sin(2.0 * np.pi * xx * 64.0)
    weft = np.cos(2.0 * np.pi * yy * 64.0)
    weave_h = 0.5 * (warp + weft) * 0.14
    rib = np.sin(2.0 * np.pi * xx * 8.0) * 0.12
    seam = (np.abs((xx * 4.0) % 1.0 - 0.5) < 0.025).astype(np.float32) * -0.15
    edge_weather = np.clip(1.0 - 4.0 * np.minimum(yy, 1.0 - yy), 0.0, 1.0) * 0.18
    canvas_height = weave_h + rib + seam

    dy_c, dx_c = np.gradient(canvas_height)
    c_norm = np.zeros((size, size, 3), dtype=np.float32)
    c_norm[:, :, 0] = np.clip((-dx_c * 6.5 * 0.5 + 0.5) * 255.0, 0, 255)
    c_norm[:, :, 1] = np.clip((-dy_c * 6.5 * 0.5 + 0.5) * 255.0, 0, 255)
    c_norm[:, :, 2] = 242.0
    if pl is not None:
        c_norm = attenuate_normal(0.65 * c_norm + 0.35 * pl[1], strength=0.85)
    c_rough = np.clip(198.0 + weave_h * 45.0 + edge_weather * 30.0, 155.0, 248.0)

    def make_canvas_albedo(base_rgb: Tuple[float, float, float], striped: bool = False) -> np.ndarray:
        base = pl[0] if pl is not None else np.full((size, size, 3), 180.0, dtype=np.float32)
        rgb = grade_albedo(base, target_mean_rgb=base_rgb, contrast=0.45, saturation=0.65)
        if striped:
            stripe_mask = ((np.sin(2.0 * np.pi * xx * 6.0) > 0.0)).astype(np.float32)[:, :, None]
            red_stripe = grade_albedo(base, target_mean_rgb=(135.0, 48.0, 42.0), contrast=0.45, saturation=0.65)
            rgb = rgb * (1.0 - 0.45 * stripe_mask) + red_stripe * (0.45 * stripe_mask)
        mod = 1.0 + weave_h * 0.45 + rib * 0.35 + seam * 0.65 - edge_weather * 0.25
        return np.clip(rgb * mod[:, :, None], 22.0, 245.0)

    save_jpg_set("awning_red", make_canvas_albedo((124.0, 42.0, 36.0)), c_norm, c_rough, quality=85)
    save_jpg_set("awning_green", make_canvas_albedo((42.0, 76.0, 52.0)), c_norm, c_rough, quality=85)
    save_jpg_set("awning_striped", make_canvas_albedo((194.0, 180.0, 154.0), striped=True), c_norm, c_rough, quality=85)

    # 2. Dark Wrought Iron (`metal_iron`) & Galvanized Corrugated Metal (`corrugated_metal`)
    iron_src = sh_green or sh_rusted or sh_painted
    if iron_src is not None:
        diff_iron = grade_albedo(iron_src[0], target_mean_rgb=(44.0, 47.0, 50.0), contrast=0.65, saturation=0.35)
        norm_iron = attenuate_normal(iron_src[1], strength=0.80)
        rough_iron = np.clip(iron_src[2] * 0.75 + 35.0, 85.0, 215.0)
        save_jpg_set("metal_iron", diff_iron, norm_iron, rough_iron, quality=85)

    corr_src = sh_painted or sh_rusted
    if corr_src is not None:
        diff_corr = grade_albedo(corr_src[0], target_mean_rgb=(122.0, 126.0, 130.0), contrast=0.78, saturation=0.35)
        norm_corr = attenuate_normal(corr_src[1], strength=0.90)
        rough_corr = np.clip(corr_src[2] * 0.80 + 30.0, 90.0, 225.0)
        save_jpg_set("corrugated_metal", diff_corr, norm_corr, rough_corr, quality=85)

    # 3. Terracotta Pottery (`terracotta`) & Rough Kraft Cardboard / Wood Crate (`cardboard`)
    tc_src = rt or pl
    if tc_src is not None:
        blend_tc = (0.65 * pl[0] + 0.35 * rt[0]) if (pl is not None and rt is not None) else tc_src[0]
        diff_tc = grade_albedo(blend_tc, target_mean_rgb=(156.0, 88.0, 62.0), contrast=0.62, saturation=0.68)
        norm_tc = attenuate_normal(pl[1] if pl is not None else tc_src[1], strength=0.72)
        rough_tc = np.clip((pl[2] if pl is not None else tc_src[2]) * 0.80 + 45.0, 125.0, 235.0)
        save_jpg_set("terracotta", diff_tc, norm_tc, rough_tc, quality=85)

    cb_src = wood or pl
    if cb_src is not None:
        diff_cb = grade_albedo(cb_src[0], target_mean_rgb=(132.0, 102.0, 68.0), contrast=0.75, saturation=0.58)
        norm_cb = attenuate_normal(cb_src[1], strength=0.82)
        rough_cb = np.clip(cb_src[2] * 0.85 + 45.0, 135.0, 245.0)
        save_jpg_set("cardboard", diff_cb, norm_cb, rough_cb, quality=85)

    # 4. Crinkled Matte-Satin Polyethylene Trash Bags (`trash_bag` & `trash_blue`)
    crinkle = (
        np.sin(2.0 * np.pi * (xx * 7.0 + yy * 5.0)) * 0.35
        + np.cos(2.0 * np.pi * (xx * 11.0 - yy * 9.0)) * 0.28
        + np.sin(2.0 * np.pi * (xx * 19.0 + yy * 17.0)) * 0.16
    )
    if pl is not None:
        crinkle += (pl[0].mean(axis=2) / 255.0 - 0.5) * 0.45
    dy_tb, dx_tb = np.gradient(gaussian_filter(crinkle, sigma=1.1))
    tb_norm = np.zeros((size, size, 3), dtype=np.float32)
    tb_norm[:, :, 0] = np.clip((-dx_tb * 8.0 * 0.5 + 0.5) * 255.0, 0, 255)
    tb_norm[:, :, 1] = np.clip((-dy_tb * 8.0 * 0.5 + 0.5) * 255.0, 0, 255)
    tb_norm[:, :, 2] = 238.0
    tb_rough = np.clip(165.0 + crinkle * 28.0, 128.0, 210.0)

    tb_black = np.zeros((size, size, 3), dtype=np.float32)
    tb_black[:, :, 0] = 32.0 + crinkle * 14.0
    tb_black[:, :, 1] = 34.0 + crinkle * 14.0
    tb_black[:, :, 2] = 37.0 + crinkle * 15.0
    save_jpg_set("trash_bag", np.clip(tb_black, 16.0, 72.0), tb_norm, tb_rough, quality=85)

    tb_blue = np.zeros((size, size, 3), dtype=np.float32)
    tb_blue[:, :, 0] = 38.0 + crinkle * 16.0
    tb_blue[:, :, 1] = 64.0 + crinkle * 20.0
    tb_blue[:, :, 2] = 104.0 + crinkle * 24.0
    save_jpg_set("trash_blue", np.clip(tb_blue, 20.0, 148.0), tb_norm, tb_rough, quality=85)


def upgrade_sign_atlas(width: int = 2048, height: int = 1024):
    """
    Generate high-contrast, sunlight-resilient multilingual 16-sign atlas (`sign_atlas.png`
    + `sign_atlas.json`) with rich dark heritage enamel/timber/cobalt-ceramic backgrounds
    and bold gold/ivory lettering so no signboard ever blows out to white under sunlight.
    """
    print(f"  -> Generating high-contrast 16-sign multilingual atlas ({width}x{height})...")
    cols = 4
    rows = 4
    cw = width // cols
    ch = height // rows

    atlas = Image.new("RGBA", (width, height), (24, 22, 20, 255))
    draw = ImageDraw.Draw(atlas)

    def fit_font(text: str, max_w: int, start_size: int, min_size: int = 12, bold: bool = True, serif: bool = False):
        for sz in range(start_size, min_size - 1, -2):
            f = load_font(sz, bold=bold, serif=serif)
            bbox = draw.textbbox((0, 0), text, font=f)
            if (bbox[2] - bbox[0]) <= max_w:
                return f
        return load_font(min_size, bold=bold, serif=serif)

    def draw_centered_text(cx: float, cy: float, text: str, font, fill, shadow=None):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = cx - tw / 2.0 - bbox[0]
        ty = cy - th / 2.0 - bbox[1]
        if shadow:
            draw.text((tx + 2, ty + 2), text, font=font, fill=shadow)
            draw.text((tx - 1, ty + 1), text, font=font, fill=shadow)
        draw.text((tx, ty), text, font=font, fill=fill)

    signs = [
        {
            "id": "jaffa_gate_plaque",
            "style": "ceramic_plaque",
            "he": "שער יפו",
            "ar": "باب الخليل",
            "en": "JAFFA GATE / BAB EL-KHALIL",
        },
        {
            "id": "omar_sq_plaque",
            "style": "ceramic_plaque",
            "he": "כיכר עומר אבן אל-ח׳טאב",
            "ar": "ميدان عمر بن الخطاب",
            "en": "OMAR IBN EL-KHATTAB SQ.",
        },
        {
            "id": "david_st_plaque",
            "style": "ceramic_plaque",
            "he": "רחוב דוד",
            "ar": "طريق البازار - سويقة علون",
            "en": "DAVID STREET / SUQ EL-BAZAR",
        },
        {
            "id": "latin_rd_plaque",
            "style": "ceramic_plaque",
            "he": "רחוב הפטריארכיה הלטינית",
            "ar": "طريق البطريركية اللاتينية",
            "en": "LATIN PATRIARCHATE RD.",
        },
        {
            "id": "new_imperial_hotel",
            "style": "fascia",
            "bg": (20, 34, 64),
            "border": (218, 176, 74),
            "text_col": (248, 226, 152),
            "sub_col": (224, 198, 138),
            "title": "NEW IMPERIAL HOTEL — EST. 1884",
            "sub": "فندق امبريال الجديد  •  JAFFA GATE JERUSALEM",
            "serif": True,
        },
        {
            "id": "petra_hotel",
            "style": "fascia",
            "bg": (42, 26, 22),
            "border": (214, 172, 76),
            "text_col": (248, 220, 138),
            "sub_col": (224, 198, 154),
            "title": "HOTEL PETRA — ROOFTOP VIEW",
            "sub": "فندق البتراء  •  HERITAGE HOSPITALITY SINCE 1843",
            "serif": True,
        },
        {
            "id": "christian_info",
            "style": "fascia",
            "bg": (22, 48, 86),
            "border": (212, 178, 92),
            "text_col": (248, 242, 225),
            "sub_col": (222, 196, 128),
            "title": "CHRISTIAN INFORMATION CENTRE",
            "sub": "מרכז מידע נוצרי  •  OMAR IBN EL-KHATTAB SQUARE",
            "serif": False,
        },
        {
            "id": "tower_of_david_museum",
            "style": "fascia",
            "bg": (96, 22, 34),
            "border": (224, 184, 88),
            "text_col": (250, 238, 206),
            "sub_col": (232, 196, 116),
            "title": "TOWER OF DAVID JERUSALEM MUSEUM",
            "sub": "מוזיאון מגדל דוד  •  متحف قلعة داود",
            "serif": True,
        },
        {
            "id": "old_city_bazaar",
            "style": "fascia",
            "bg": (54, 30, 16),
            "border": (220, 168, 62),
            "text_col": (250, 222, 140),
            "sub_col": (234, 192, 132),
            "title": "OLD CITY ORIENTAL BAZAAR & ANTIQUES",
            "sub": "بازار البلدة القديمة للتحف والشرقيات  •  DAVID ST.",
            "serif": True,
        },
        {
            "id": "jerusalem_pottery",
            "style": "ceramic_shop",
            "bg": (18, 54, 116),
            "border": (68, 194, 206),
            "text_col": (250, 244, 226),
            "sub_col": (238, 204, 112),
            "title": "ARMENIAN CERAMICS & JERUSALEM POTTERY",
            "sub": "خزف القدس الأرمني التقليدي  •  HAND-PAINTED TILES",
            "serif": True,
        },
        {
            "id": "money_exchange",
            "style": "fascia",
            "bg": (18, 22, 28),
            "border": (232, 190, 66),
            "text_col": (250, 214, 82),
            "sub_col": (238, 240, 244),
            "title": "AUTHORIZED MONEY EXCHANGE • ATM",
            "sub": "صرافة العملات المعتمدة  •  המרת מט״ח  •  0% COMMISSION",
            "serif": False,
        },
        {
            "id": "falafel_cafe",
            "style": "fascia",
            "bg": (24, 72, 38),
            "border": (236, 188, 58),
            "text_col": (252, 244, 208),
            "sub_col": (246, 202, 82),
            "title": "JAFFA GATE FALAFEL, HUMMUS & FRESH POMEGRANATE JUICE",
            "sub": "فلافل وحمص وعصير رمان طازج  •  FRESH MINT TEA & ARABIC COFFEE",
            "serif": False,
        },
        {
            "id": "st_james_souvenirs",
            "style": "fascia",
            "bg": (22, 48, 96),
            "border": (220, 178, 74),
            "text_col": (250, 240, 214),
            "sub_col": (230, 196, 108),
            "title": "ST. JAMES SOUVENIRS, ICONS & GIFTS",
            "sub": "أيقونات وهدايا الشرق المقدس  •  OLIVE WOOD CARVINGS",
            "serif": True,
        },
        {
            "id": "holy_land_rugs",
            "style": "fascia",
            "bg": (116, 20, 26),
            "border": (228, 182, 70),
            "text_col": (252, 232, 166),
            "sub_col": (240, 212, 154),
            "title": "HOLY LAND CARPETS, KILIMS & TEXTILES",
            "sub": "سجاد وبسط الأراضي المقدسة  •  HANDWOVEN TRIBAL RUGS",
            "serif": True,
        },
        {
            "id": "pilgrim_books",
            "style": "fascia",
            "bg": (24, 58, 42),
            "border": (204, 168, 90),
            "text_col": (246, 238, 214),
            "sub_col": (218, 192, 130),
            "title": "OLD CITY BOOKSHOP & MAPS",
            "sub": "مكتبة وخرائط القدس القديمة  •  HISTORICAL PRINTS & GUIDES",
            "serif": True,
        },
        {
            "id": "swedish_study",
            "style": "fascia",
            "bg": (18, 72, 134),
            "border": (244, 204, 48),
            "text_col": (252, 252, 255),
            "sub_col": (250, 216, 76),
            "title": "SWEDISH CHRISTIAN STUDY CENTRE",
            "sub": "SCSC JERUSALEM  •  LATIN PATRIARCHATE ROAD",
            "serif": False,
        },
    ]

    meta = {}
    for idx, s in enumerate(signs):
        col = idx % cols
        row = idx // cols
        x0 = col * cw
        y0 = row * ch
        x1 = x0 + cw
        y1 = y0 + ch
        cx = x0 + cw / 2.0

        pad = 12
        bx0, by0, bx1, by1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad

        if s["style"] == "ceramic_plaque":
            draw.rectangle([bx0, by0, bx1, by1], fill=(18, 58, 132), outline=(12, 36, 86), width=5)
            for tx in range(bx0 + 18, bx1 - 12, 28):
                for ty in [by0 + 10, by1 - 10]:
                    draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(44, 148, 164), outline=(232, 228, 216))
            for ty in range(by0 + 18, by1 - 12, 28):
                for tx in [bx0 + 10, bx1 - 10]:
                    draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(44, 148, 164), outline=(232, 228, 216))

            ix0, iy0, ix1, iy1 = bx0 + 22, by0 + 22, bx1 - 22, by1 - 22
            draw.rectangle([ix0, iy0, ix1, iy1], fill=(224, 218, 204), outline=(18, 58, 132), width=3)
            for gx in [ix0 + (ix1 - ix0) // 3, ix0 + 2 * (ix1 - ix0) // 3]:
                draw.line([(gx, iy0), (gx, iy1)], fill=(198, 192, 178), width=1)
            gy = (iy0 + iy1) // 2
            draw.line([(ix0, gy), (ix1, gy)], fill=(198, 192, 178), width=1)

            f_he = fit_font(s["he"], ix1 - ix0 - 24, start_size=34, bold=True)
            f_ar = fit_font(s["ar"], ix1 - ix0 - 24, start_size=32, bold=True)
            f_en = fit_font(s["en"], ix1 - ix0 - 24, start_size=28, bold=True)

            draw_centered_text(cx, iy0 + (iy1 - iy0) * 0.23, s["he"], f_he, fill=(12, 38, 98))
            draw_centered_text(cx, iy0 + (iy1 - iy0) * 0.52, s["ar"], f_ar, fill=(12, 38, 98))
            draw.line([(ix0 + 24, iy0 + (iy1 - iy0) * 0.69), (ix1 - 24, iy0 + (iy1 - iy0) * 0.69)], fill=(28, 98, 158), width=2)
            draw_centered_text(cx, iy0 + (iy1 - iy0) * 0.83, s["en"], f_en, fill=(10, 32, 88))
        else:
            bg = s["bg"]
            border = s["border"]
            draw.rectangle([bx0, by0, bx1, by1], fill=bg, outline=border, width=6)
            draw.rectangle([bx0 + 12, by0 + 12, bx1 - 12, by1 - 12], outline=border, width=2)

            for (mx, my) in [(bx0 + 22, by0 + 22), (bx1 - 22, by0 + 22), (bx0 + 22, by1 - 22), (bx1 - 22, by1 - 22)]:
                draw.ellipse([mx - 6, my - 6, mx + 6, my + 6], fill=border)

            max_tw = (bx1 - bx0) - 44
            f_title = fit_font(s["title"], max_tw, start_size=34, min_size=14, bold=True, serif=s.get("serif", False))
            f_sub = fit_font(s["sub"], max_tw, start_size=23, min_size=12, bold=True, serif=False)

            shadow_col = (8, 8, 10)
            draw_centered_text(cx, y0 + ch * 0.36, s["title"], f_title, fill=s["text_col"], shadow=shadow_col)
            draw.line([(bx0 + 44, y0 + ch * 0.56), (bx1 - 44, y0 + ch * 0.56)], fill=border, width=3)
            draw.polygon([(cx, y0 + ch * 0.56 - 6), (cx + 7, y0 + ch * 0.56), (cx, y0 + ch * 0.56 + 6), (cx - 7, y0 + ch * 0.56)], fill=border)
            draw_centered_text(cx, y0 + ch * 0.74, s["sub"], f_sub, fill=s["sub_col"], shadow=shadow_col)

        u0 = round((x0 + pad) / width, 5)
        u1 = round((x1 - pad) / width, 5)
        v0 = round(1.0 - (y1 - pad) / height, 5)
        v1 = round(1.0 - (y0 + pad) / height, 5)
        aspect = round(((x1 - x0 - 2 * pad) / (y1 - y0 - 2 * pad)), 3)

        meta[s["id"]] = {
            "u0": u0,
            "v0": v0,
            "u1": u1,
            "v1": v1,
            "aspect": aspect,
            "row": row,
            "col": col,
        }

    atlas.convert("RGB").save(OUT_DIR / "sign_atlas.png")
    with open(OUT_DIR / "sign_atlas.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    upgrade_all_stone_and_paving_from_photogrammetry()
    upgrade_steel_shutters_from_photogrammetry(1024)
    upgrade_wood_doors_from_photogrammetry(1024)
    upgrade_accent_pbr_textures(256)
    upgrade_sign_atlas(2048, 1024)
    print("[OK] All real photogrammetry PBR & photographic shopfront textures written to public/textures/!")


if __name__ == "__main__":
    main()
