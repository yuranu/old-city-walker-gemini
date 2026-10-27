#!/usr/bin/env python3
"""
Clean-Room Reference Acquisition Pipeline for Jaffa Gate District, Jerusalem.

1. Writes the 10 canonical viewpoints to `data/places.json` and `public/data/places.json`.
2. Queries Wikimedia Commons API (`https://commons.wikimedia.org/w/api.php`) for real
   street-level outdoor architectural photographs of each viewpoint around Jaffa Gate,
   Tower of David, Omar Ibn El-Khattab Square, New Imperial Hotel, Petra Hotel,
   Latin Patriarchate Road, and David Street souq.
3. Validates every candidate photograph (resolution, aspect ratio, RGB outdoor colour/stone
   characteristics, and filename/category metadata filter to reject indoor museum rooms,
   maps, diagrams, scans, portraits, or tiny icons).
4. Saves validated 10 reference images to `data/references/<id>.jpg` and
   `public/references/<id>.jpg`, and writes metadata to `public/data/references.json`
   and `data/references.json`.
"""

import io
import json
import math
import time
from pathlib import Path
import requests
from PIL import Image, ImageStat

LAT0 = 31.77661
LON0 = 35.22768
ALT0 = 766.0

COS_LAT0 = math.cos(math.radians(LAT0))
METERS_PER_DEG_LON = 111320.0 * COS_LAT0
METERS_PER_DEG_LAT = 110540.0

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "OldCityWalkerCleanRoom/1.0 (https://commons.wikimedia.org; educational 3D architecture reconstruction)"

PLACES = [
    {
        "id": "west_plaza",
        "name": "Jaffa Gate West Approach Plaza",
        "x": -34.0,
        "y": 5.0,
        "z_eye": 1.7,
        "heading": 98.0,
        "pitch": 4.0,
        "fov": 70.0,
        "radius": 30.0,
    },
    {
        "id": "jaffa_gate_portal",
        "name": "Jaffa Gate (Bab el-Khalil) Outer Portal",
        "x": -16.0,
        "y": 2.0,
        "z_eye": 1.7,
        "heading": 94.0,
        "pitch": 8.0,
        "fov": 70.0,
        "radius": 20.0,
    },
    {
        "id": "jaffa_gate_passage",
        "name": "Jaffa Gate Vaulted L-Passage",
        "x": -1.5,
        "y": 4.5,
        "z_eye": 1.7,
        "heading": 145.0,
        "pitch": 5.0,
        "fov": 75.0,
        "radius": 15.0,
    },
    {
        "id": "breach_1898",
        "name": "1898 Wall Breach & Moat Road",
        "x": 6.0,
        "y": -10.0,
        "z_eye": 1.7,
        "heading": 125.0,
        "pitch": 4.0,
        "fov": 70.0,
        "radius": 25.0,
    },
    {
        "id": "tower_of_david",
        "name": "Tower of David (Phasael Tower) & Citadel Moat",
        "x": 24.0,
        "y": -16.0,
        "z_eye": 1.7,
        "heading": 170.0,
        "pitch": 8.0,
        "fov": 70.0,
        "radius": 45.0,
    },
    {
        "id": "omar_square_west",
        "name": "Omar Ibn El-Khattab Square (Looking East)",
        "x": 22.0,
        "y": -2.5,
        "z_eye": 1.7,
        "heading": 85.0,
        "pitch": 4.0,
        "fov": 70.0,
        "radius": 35.0,
    },
    {
        "id": "omar_square_north",
        "name": "New Imperial Hotel & Petra Hotel North Facades",
        "x": 46.0,
        "y": -9.0,
        "z_eye": 1.7,
        "heading": 15.0,
        "pitch": 8.0,
        "fov": 70.0,
        "radius": 30.0,
    },
    {
        "id": "omar_square_east",
        "name": "Omar Square East (Looking Back to Jaffa Gate)",
        "x": 76.0,
        "y": 2.0,
        "z_eye": 1.7,
        "heading": 265.0,
        "pitch": 5.0,
        "fov": 70.0,
        "radius": 30.0,
    },
    {
        "id": "latin_patriarchate_lane",
        "name": "Latin Patriarchate Road",
        "x": 36.0,
        "y": 16.0,
        "z_eye": 1.7,
        "heading": 352.0,
        "pitch": 3.0,
        "fov": 70.0,
        "radius": 25.0,
    },
    {
        "id": "david_street_souq",
        "name": "David Street Stepped Souq Entrance",
        "x": 90.0,
        "y": 5.5,
        "z_eye": 1.7,
        "heading": 88.0,
        "pitch": 2.0,
        "fov": 70.0,
        "radius": 30.0,
    },
]

# Curated primary Wikimedia Commons titles + search fallbacks for each of the 10 viewpoints
PREFERRED_COMMONS_TITLES = {
    "west_plaza": [
        "File:Jerusalem Jaffa Gate BW 1.JPG",
        "File:Jaffa Gate, Jerusalem 09.jpg",
        "File:Jaffa Gate Jerusalem.jpg",
    ],
    "jaffa_gate_portal": [
        "File:Jerusalem Jaffa Gate BW 2010-09-20 12-21-17.JPG",
        "File:Jaffa Gate IMG 1498.jpg",
        "File:146240 jaffa gate PikiWiki Israel.jpg",
    ],
    "jaffa_gate_passage": [
        "File:Jerusalem Wikimania Tour P1040497.JPG",
        "File:127937 jerusalem - jaffa gate - in the spirit of the time PikiWiki Israel.jpg",
    ],
    "breach_1898": [
        "File:The Jaffa Gate (3620435666).jpg",
        "File:Old City Walls and Tower of David in Jerusalem (2019) 03.jpg",
    ],
    "tower_of_david": [
        "File:Old City Walls and Tower of David in Jerusalem (2019) 01.jpg",
        "File:Jerusalem-Tower-of-David-V2-348.jpg",
        "File:DSCF5104 Tower of David.jpg",
    ],
    "omar_square_west": [
        "File:Omar Ibn El-Khattab Square.jpg",
        "File:Omar Ibn El-Khattab Square, 2019 (01).jpg",
    ],
    "omar_square_north": [
        "File:New Imperial hotel 3 - Jerusalem.jpg",
        "File:Jerusalem-Imperial-Hotel-1724.jpg",
    ],
    "omar_square_east": [
        "File:Jerusalem-Petra-Hotel-MKK-085.jpg",
        "File:Omar Ibn El-Khattab Square, 2019 (03).jpg",
        "File:16-03-30-Ста́рый го́род Иерусали́ма-RalfR-DSCF7641.jpg",
    ],
    "latin_patriarchate_lane": [
        "File:Jerusalem Doorway on Latin Patriarchate Road.jpg",
        "File:Omar Ben El Hatab street, Jerusalem.JPG",
        "File:Latin Patriarchate IMG 1-3000-215.jpg",
    ],
    "david_street_souq": [
        "File:16-03-30-Ста́рый го́род Иерусали́ма-RalfR-DSCF7642.jpg",
        "File:147292 jerusalem old city - david street PikiWiki Israel.jpg",
        "File:Davidstreet.jpg",
    ],
}

PLACE_SEARCH_QUERIES = {
    "west_plaza": ["Jaffa Gate Jerusalem", "Jaffa Gate exterior plaza"],
    "jaffa_gate_portal": ["Jaffa Gate portal Jerusalem", "Bab el-Khalil Jaffa Gate"],
    "jaffa_gate_passage": ["Jaffa Gate interior passage Jerusalem", "Jaffa Gate archway"],
    "breach_1898": ["Jaffa Gate road Tower of David", "Jaffa Gate wall road"],
    "tower_of_david": ["Tower of David Jerusalem", "Citadel of Jerusalem moat"],
    "omar_square_west": ["Omar Ibn El-Khattab Square", "Omar Square Jaffa Gate"],
    "omar_square_north": ["Imperial Hotel Jerusalem", "New Imperial Hotel Jerusalem"],
    "omar_square_east": ["Petra Hotel Jerusalem", "Omar Ibn El-Khattab Square"],
    "latin_patriarchate_lane": ["Latin Patriarchate Road Jerusalem", "Latin Patriarchate Street"],
    "david_street_souq": ["David Street Jerusalem", "David street old city jerusalem"],
}

REJECT_WORDS = (
    "map",
    "plan",
    "diagram",
    "drawing",
    "engraving",
    "lithograph",
    "painting",
    "stamp",
    "coin",
    "medal",
    "logo",
    "icon",
    "coat of arms",
    "exhibit",
    "exhibition",
    "museum interior",
    "glass case",
    "artifact",
    "pottery",
    "manuscript",
    "portrait",
    "flag",
    "svg",
    "pdf",
    "ogg",
    "webm",
    "tif",
    "nypl",
    "circa 19",
    "1898_",
    "1900",
    "1917",
    "1920",
    "1930",
    "black and white",
    "b&w",
    "mattson",
)


def xy_to_ll(x: float, y: float) -> tuple[float, float]:
    lon = LON0 + x / METERS_PER_DEG_LON
    lat = LAT0 + y / METERS_PER_DEG_LAT
    return round(lat, 6), round(lon, 6)


def fetch_commons_titles_info(session: requests.Session, titles: list[str]) -> list[dict]:
    if not titles:
        return []
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1280,
        "format": "json",
    }
    out = []
    try:
        resp = session.get(COMMONS_API, params=params, timeout=20)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        by_title = {}
        for _, page in pages.items():
            t = page.get("title", "")
            ii_list = page.get("imageinfo", [])
            if ii_list:
                by_title[t] = ii_list[0]
        for t in titles:
            if t in by_title:
                out.append({"title": t, "info": by_title[t], "query": "curated_title"})
    except Exception as exc:
        print(f"[Commons] Title lookup warning: {exc}")
    return out


def search_commons_candidates(
    session: requests.Session, preferred_titles: list[str], queries: list[str], lat: float, lon: float
) -> list[dict]:
    candidates = []
    seen_titles = set()

    # 1. Preferred verified Commons titles first
    for item in fetch_commons_titles_info(session, preferred_titles):
        seen_titles.add(item["title"])
        candidates.append(item)

    # 2. Text search using list=search to preserve relevance order
    for q in queries:
        try:
            s_resp = session.get(
                COMMONS_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": f"filetype:bitmap {q}",
                    "srnamespace": 6,
                    "srlimit": 10,
                    "format": "json",
                },
                timeout=20,
            )
            s_resp.raise_for_status()
            titles = [
                x["title"]
                for x in s_resp.json().get("query", {}).get("search", [])
                if x["title"] not in seen_titles
            ]
            for item in fetch_commons_titles_info(session, titles):
                seen_titles.add(item["title"])
                item["query"] = q
                candidates.append(item)
        except Exception as exc:
            print(f"[Commons] Search warning for '{q}': {exc}")

    return candidates


def is_valid_architectural_photo(title: str, info: dict) -> bool:
    mime = info.get("mime", "")
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        return False

    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    if width < 600 or height < 400:
        return False

    aspect = width / max(1, height)
    if aspect < 0.6 or aspect > 2.6:
        return False

    ext = info.get("extmetadata", {})
    desc = str(ext.get("ImageDescription", {}).get("value", "")).lower()
    cats = str(ext.get("Categories", {}).get("value", "")).lower()
    combined_text = f"{title.lower()} {desc} {cats}"

    for bad in REJECT_WORDS:
        if bad in combined_text:
            return False

    return True


def verify_and_save_image_bytes(raw_bytes: bytes, dest_paths: list[Path]) -> tuple[bool, dict]:
    """
    Open image with Pillow, verify it is a full-color outdoor/architectural photograph
    (not grayscale archival scan, not dark indoor room, not solid icon), resize to max 1280px,
    and save as high-quality JPEG.
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img = img.convert("RGB")
        w, h = img.size
        if w < 500 or h < 350:
            return False, {}

        stat = ImageStat.Stat(img)
        mean_r, mean_g, mean_b = stat.mean
        std_r, std_g, std_b = stat.stddev

        brightness = (mean_r + mean_g + mean_b) / 3.0
        avg_std = (std_r + std_g + std_b) / 3.0

        if brightness < 45.0 or brightness > 235.0:
            return False, {}
        if avg_std < 28.0:
            return False, {}

        small = img.resize((64, 64))
        raw_rgb = small.tobytes()
        color_diffs = [
            max(raw_rgb[i], raw_rgb[i + 1], raw_rgb[i + 2])
            - min(raw_rgb[i], raw_rgb[i + 1], raw_rgb[i + 2])
            for i in range(0, len(raw_rgb), 3)
        ]
        mean_sat = sum(color_diffs) / len(color_diffs)
        if mean_sat < 10.0:
            return False, {}

        if w > 1280 or h > 1280:
            img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            w, h = img.size

        for p in dest_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p, format="JPEG", quality=88, optimize=True)

        return True, {
            "width": w,
            "height": h,
            "mean_brightness": round(brightness, 1),
            "mean_saturation": round(mean_sat, 1),
            "bytes": dest_paths[0].stat().st_size,
        }
    except Exception as exc:
        return False, {"error": str(exc)}


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    pub_data_dir = root / "public" / "data"
    ref_dir = data_dir / "references"
    pub_ref_dir = root / "public" / "references"

    for d in (data_dir, pub_data_dir, ref_dir, pub_ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Preserve full 29-POI data/places.json if present, and sync to public/data/places.json
    src_places = data_dir / "places.json"
    if src_places.exists():
        places_out = json.loads(src_places.read_text(encoding="utf-8"))
    else:
        places_out = [dict(pl) for pl in PLACES]
        src_places.write_text(json.dumps(places_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (pub_data_dir / "places.json").write_text(
        json.dumps(places_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] Synced public/data/places.json ({len(places_out)} viewpoints)")

    # 2. Fetch distinct, verified Wikimedia Commons photographs for each of the 10 viewpoints
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    used_titles = set()
    references_manifest = []

    for pl in PLACES:
        pid = pl["id"]
        lat, lon = xy_to_ll(pl["x"], pl["y"])
        preferred = PREFERRED_COMMONS_TITLES.get(pid, [])
        queries = PLACE_SEARCH_QUERIES.get(pid, [f"{pl['name']} Jerusalem"])
        print(f"\n[Reference] Acquiring outdoor architectural photo for '{pid}' ({pl['name']})...")

        candidates = search_commons_candidates(session, preferred, queries, lat, lon)
        saved_meta = None

        for cand in candidates:
            title = cand["title"]
            if title in used_titles:
                continue
            info = cand["info"]
            if not is_valid_architectural_photo(title, info):
                continue

            img_url = info.get("thumburl") or info.get("url")
            if not img_url:
                continue

            try:
                resp = session.get(img_url, timeout=25)
                resp.raise_for_status()
                dest_files = [ref_dir / f"{pid}.jpg", pub_ref_dir / f"{pid}.jpg"]
                ok, img_stats = verify_and_save_image_bytes(resp.content, dest_files)
                if ok:
                    import re

                    used_titles.add(title)
                    ext = info.get("extmetadata", {})
                    raw_artist = str(ext.get("Artist", {}).get("value", "") or "")
                    artist = re.sub(r"<[^>]+>", "", raw_artist).strip() or "Wikimedia Commons contributor"
                    lic_short = str(ext.get("LicenseShortName", {}).get("value", "") or "CC BY-SA").strip()
                    saved_meta = {
                        "id": pid,
                        "name": pl["name"],
                        "x": pl["x"],
                        "y": pl["y"],
                        "z_eye": pl["z_eye"],
                        "heading": pl["heading"],
                        "pitch": pl["pitch"],
                        "fov": pl["fov"],
                        "lat": lat,
                        "lon": lon,
                        "image": f"/references/{pid}.jpg",
                        "local_path": f"data/references/{pid}.jpg",
                        "source": "Wikimedia Commons",
                        "commons_title": title,
                        "author": artist,
                        "license": lic_short,
                        "source_url": info.get("descriptionurl", img_url),
                        "stats": img_stats,
                    }
                    print(
                        f"  -> Saved {pid}.jpg from '{title}' "
                        f"({img_stats['width']}x{img_stats['height']}, {img_stats['bytes']:,} bytes)"
                    )
                    break
            except Exception as exc:
                print(f"  -> Download warning for {title}: {exc}")
            time.sleep(0.3)

        if not saved_meta:
            raise RuntimeError(f"Could not find a valid outdoor architectural photo for {pid}")

        references_manifest.append(saved_meta)

    for p in (data_dir / "references.json", pub_data_dir / "references.json"):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(references_manifest, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Wrote {p} ({len(references_manifest)} reference entries)")


if __name__ == "__main__":
    main()
