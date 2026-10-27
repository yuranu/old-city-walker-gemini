#!/usr/bin/env python3
"""
Fetches real Wikimedia Commons reference photographs for all 19 new Old City POIs
(Gates, Temples, and Touristic Spots) and updates `public/references/<id>.jpg`
and `public/data/references.json`.
"""

import json
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
PUB_REF_DIR = ROOT_DIR / "public" / "references"
DATA_REF_DIR = ROOT_DIR / "data" / "references"
REF_JSON_PATH = ROOT_DIR / "public" / "data" / "references.json"

POI_QUERIES = {
    "holy_sepulchre_parvis": "Church of the Holy Sepulchre facade courtyard Jerusalem",
    "muristan_redeemer": "Lutheran Church of the Redeemer Jerusalem Muristan",
    "new_gate_portal": "New Gate Jerusalem Old City",
    "triple_souq_cardo": "Old City market Jerusalem bazaar",
    "chain_street_turba": "Bab al-Silsila Jerusalem",
    "st_james_cathedral": "Cathedral of Saint James Jerusalem Armenian Quarter",
    "zion_gate_portal": "Zion Gate Jerusalem",
    "hurva_synagogue_plaza": "Hurva Synagogue Jerusalem exterior",
    "southern_roman_cardo": "Cardo Jerusalem columns",
    "western_wall_plaza": "Western Wall Plaza Jerusalem Kotel",
    "dung_gate_portal": "Dung Gate Jerusalem",
    "dome_of_the_rock_south": "Dome of the Rock Jerusalem Qubbat al-Sakhra",
    "dome_of_the_chain": "Dome of the Chain Jerusalem",
    "al_aqsa_mosque_porch": "Al-Aqsa Mosque facade Jerusalem",
    "golden_gate_sanctuary": "Golden Gate Jerusalem Temple Mount",
    "damascus_gate_plaza": "Damascus Gate Jerusalem",
    "ecce_homo_via_dolorosa": "Ecce Homo arch Via Dolorosa Jerusalem",
    "st_anne_and_lions_gate": "Church of Saint Anne Jerusalem facade",
    "herods_gate_portal": "Herod's Gate Jerusalem",
}

HEADERS = {"User-Agent": "OldCityWalkerBot/1.0 (https://old-city-walker-gemini.yuranu.com)"}


def fetch_commons_photo(query: str):
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": "6",
        "prop": "imageinfo",
        "iiprop": "url|dimensions|extmetadata",
        "iiurlwidth": "960",
        "format": "json",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    pages = data.get("query", {}).get("pages", {})
    for _, page in sorted(pages.items(), key=lambda kv: kv[1].get("index", 99)):
        ii_list = page.get("imageinfo", [])
        if not ii_list:
            continue
        ii = ii_list[0]
        w = int(ii.get("width", 0))
        h = int(ii.get("height", 0))
        if w < 500 or h < 350 or w < h * 0.65:
            continue
        thumb_url = ii.get("thumburl") or ii.get("url")
        if not thumb_url:
            continue
        try:
            img_req = urllib.request.Request(thumb_url, headers=HEADERS)
            with urllib.request.urlopen(img_req, timeout=20) as img_resp:
                raw = img_resp.read()
            im = Image.open(BytesIO(raw)).convert("RGB")
            im = im.resize((960, 600), Image.Resampling.LANCZOS)
            import re

            ext = ii.get("extmetadata", {})
            raw_artist = str(ext.get("Artist", {}).get("value", "") or "")
            artist = re.sub(r"<[^>]+>", "", raw_artist).strip() or "Wikimedia Commons contributor"
            lic_short = str(ext.get("LicenseShortName", {}).get("value", "") or "CC BY-SA").strip()
            return im, page.get("title", ""), ii.get("descriptionurl", ""), artist, lic_short
        except Exception:
            continue
    return None, "", "", "", ""


def main():
    PUB_REF_DIR.mkdir(parents=True, exist_ok=True)
    DATA_REF_DIR.mkdir(parents=True, exist_ok=True)
    refs_meta = {}
    for candidate_path in (REF_JSON_PATH, ROOT_DIR / "data" / "references.json"):
        if candidate_path.exists():
            try:
                loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    refs_meta.update(loaded)
                elif isinstance(loaded, list):
                    for item in loaded:
                        if isinstance(item, dict) and "id" in item:
                            refs_meta[item["id"]] = item
            except Exception:
                pass

    for pid, q in POI_QUERIES.items():
        out_pub = PUB_REF_DIR / f"{pid}.jpg"
        out_dat = DATA_REF_DIR / f"{pid}.jpg"
        if out_pub.exists() and out_pub.stat().st_size > 10000 and pid in refs_meta and "author" in refs_meta[pid]:
            print(f"[SKIP] {pid}.jpg already exists")
            continue
        im, title, desc_url, artist, lic_short = fetch_commons_photo(q)
        if im is not None:
            im.save(out_pub, quality=84, optimize=True)
            im.save(out_dat, quality=84, optimize=True)
            print(f"[OK] Downloaded {pid}.jpg ({title} | {artist} | {lic_short})")
            refs_meta[pid] = {
                "id": pid,
                "image": f"/references/{pid}.jpg",
                "source": "Wikimedia Commons",
                "commons_title": title,
                "author": artist,
                "license": lic_short,
                "source_url": desc_url,
            }
        else:
            print(f"[WARN] Could not fetch photo for {pid}")

    out_text = json.dumps(refs_meta, indent=2, ensure_ascii=False)
    REF_JSON_PATH.write_text(out_text, encoding="utf-8")
    (ROOT_DIR / "data" / "references.json").write_text(out_text, encoding="utf-8")


if __name__ == "__main__":
    main()
