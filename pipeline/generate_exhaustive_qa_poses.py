#!/usr/bin/env python3
"""
Generates area-by-area multi-angle + randomized QA pose sets for:
1. `landmarks_360`: 360-degree N/E/S/W surround views of major landmarks (Al-Aqsa Mosque, Holy Sepulchre,
   Lutheran Church of the Redeemer, St. Anne Church, Austrian Hospice, St. James Cathedral,
   Hurva Synagogue, Tiferet Yisrael, Four Sephardic Synagogues, Mamluk Turbas).
2. `streets_and_exits`: Street-level and randomized off-corridor viewpoints exiting Via Dolorosa,
   Khan el-Zeit, El-Wad, Christian Quarter Rd, Armenian Patriarchate Rd, and Jewish Quarter streets.
"""

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LANDMARKS_360 = [
    # Al-Aqsa Mosque (N, E, W, S)
    {"id": "aqsa_1_north_porch", "name": "Al-Aqsa North Porch & Clerestory", "x": 762.0, "y": 29.0, "heading": 180.0, "pitch": 6.0},
    {"id": "aqsa_2_east_basilica", "name": "Al-Aqsa East Basilica Facade & Clerestory", "x": 815.0, "y": -44.0, "heading": 268.0, "pitch": 8.0},
    {"id": "aqsa_3_west_basilica", "name": "Al-Aqsa West Basilica Facade & Clerestory", "x": 708.0, "y": -44.0, "heading": 88.0, "pitch": 8.0},
    {"id": "aqsa_4_south_qibla", "name": "Al-Aqsa South Qibla & Mihrab Facade", "x": 762.0, "y": -106.0, "heading": 0.0, "pitch": 10.0},
    # Lutheran Church of the Redeemer (W, S, E, N)
    {"id": "redeemer_1_west_belfry", "name": "Redeemer West Facade & Bell Tower", "x": 222.0, "y": 124.0, "heading": 72.0, "pitch": 14.0},
    {"id": "redeemer_2_south_muristan", "name": "Redeemer South Muristan Facade", "x": 258.0, "y": 118.0, "heading": 2.0, "pitch": 12.0},
    {"id": "redeemer_3_east_apse", "name": "Redeemer East Romanesque Apse", "x": 308.0, "y": 141.5, "heading": 270.0, "pitch": 12.0},
    {"id": "redeemer_4_north_cloister", "name": "Redeemer North Facade & Cloister Portal", "x": 258.0, "y": 172.0, "heading": 180.0, "pitch": 12.0},
    # Church of the Holy Sepulchre & Mosque of Omar (S, E, N)
    {"id": "sepulchre_1_south_parvis", "name": "Holy Sepulchre South Parvis & Portals", "x": 197.0, "y": 154.0, "heading": 0.0, "pitch": 14.0},
    {"id": "sepulchre_2_east_facade", "name": "Holy Sepulchre East Basilica Facade", "x": 244.0, "y": 195.0, "heading": 270.0, "pitch": 10.0},
    {"id": "sepulchre_3_north_facade", "name": "Holy Sepulchre North Facade & Domes", "x": 197.0, "y": 234.0, "heading": 180.0, "pitch": 10.0},
    {"id": "mosque_omar_east", "name": "Mosque of Omar East Facade", "x": 224.0, "y": 166.0, "heading": 265.0, "pitch": 10.0},
    # Crusader Church of St. Anne (W, S, E, N)
    {"id": "st_anne_1_west_portal", "name": "St. Anne West Crusader Facade", "x": 808.0, "y": 526.0, "heading": 82.0, "pitch": 8.0},
    {"id": "st_anne_2_south_aisle", "name": "St. Anne South Aisle & Clerestory", "x": 835.0, "y": 502.0, "heading": 0.0, "pitch": 10.0},
    {"id": "st_anne_3_east_apses", "name": "St. Anne East Tripartite Crusader Apses", "x": 864.0, "y": 528.7, "heading": 270.0, "pitch": 10.0},
    {"id": "st_anne_4_north_aisle", "name": "St. Anne North Aisle & Clerestory", "x": 835.0, "y": 554.0, "heading": 180.0, "pitch": 10.0},
    # Austrian Hospice (W, S, E)
    {"id": "hospice_1_west_entrance", "name": "Austrian Hospice West Portal & Balcony", "x": 422.0, "y": 405.0, "heading": 68.0, "pitch": 8.0},
    {"id": "hospice_2_south_dolorosa", "name": "Austrian Hospice South Via Dolorosa Facade", "x": 448.0, "y": 386.0, "heading": 0.0, "pitch": 12.0},
    {"id": "hospice_3_east_facade", "name": "Austrian Hospice East Facade", "x": 474.0, "y": 412.0, "heading": 270.0, "pitch": 12.0},
    # St. James Cathedral & Hurva / Tiferet Yisrael (S, E, W)
    {"id": "st_james_1_south", "name": "St. James Cathedral South Portal", "x": 132.5, "y": -252.0, "heading": 0.0, "pitch": 12.0},
    {"id": "st_james_2_east", "name": "St. James Cathedral East Facade", "x": 162.0, "y": -225.0, "heading": 270.0, "pitch": 10.0},
    {"id": "hurva_1_north_plaza", "name": "Hurva Synagogue North Plaza & Menorah", "x": 346.7, "y": -105.0, "heading": 180.0, "pitch": 14.0},
    {"id": "hurva_2_south_facade", "name": "Hurva Synagogue South & East Facades", "x": 368.0, "y": -166.0, "heading": 325.0, "pitch": 14.0},
    {"id": "tiferet_east_facade", "name": "Tiferet Yisrael East & South Facade", "x": 514.0, "y": -165.0, "heading": 270.0, "pitch": 12.0},
]

STREETS_AND_EXITS = [
    # Exiting Via Dolorosa into Muslim Quarter side streets (Previously missing chunks!)
    {"id": "exit_vd_1_el_wad_south", "name": "Exiting Via Dolorosa South down El-Wad Rd", "x": 443.3, "y": 376.3, "heading": 159.6, "pitch": 3.0},
    {"id": "exit_vd_2_el_wad_north", "name": "Exiting Via Dolorosa North up El-Wad Rd", "x": 387.5, "y": 438.5, "heading": 302.0, "pitch": 3.0},
    {"id": "exit_vd_3_ecce_homo_south", "name": "Side Street South of Ecce Homo Arch", "x": 560.0, "y": 380.7, "heading": 171.0, "pitch": 3.0},
    {"id": "exit_vd_4_ecce_homo_north", "name": "Side Street North of Ecce Homo Arch", "x": 555.0, "y": 456.5, "heading": 341.0, "pitch": 3.0},
    {"id": "exit_vd_5_bab_hutta_north", "name": "Bab Hutta Street North of Eastern Via Dolorosa", "x": 745.0, "y": 495.0, "heading": 0.0, "pitch": 3.0},
    {"id": "exit_vd_6_ghawanima_south", "name": "Alley South of Eastern Via Dolorosa", "x": 745.0, "y": 435.0, "heading": 180.0, "pitch": 3.0},
    # Khan el-Zeit & Damascus Gate Inner Fork
    {"id": "street_khan_el_zeit_upper", "name": "Khan el-Zeit Souq South of Damascus Gate", "x": 275.0, "y": 440.0, "heading": 171.0, "pitch": 2.0},
    {"id": "street_khan_el_zeit_mid", "name": "Khan el-Zeit Souq Mid-Corridor", "x": 292.0, "y": 330.0, "heading": 173.0, "pitch": 2.0},
    {"id": "street_aqabat_saraya", "name": "Aqabat al-Saraya Street (Muslim Quarter)", "x": 385.0, "y": 295.0, "heading": 85.0, "pitch": 3.0},
    {"id": "street_khalidiya", "name": "Aqabat al-Khalidiya Street", "x": 365.0, "y": 185.0, "heading": 90.0, "pitch": 3.0},
    # Christian Quarter Secondary Streets
    {"id": "street_christian_qtr_rd", "name": "Christian Quarter Road North", "x": 117.0, "y": 195.0, "heading": 14.0, "pitch": 3.0},
    {"id": "street_st_francis", "name": "St. Francis Street (Christian Quarter)", "x": 0.0, "y": 285.0, "heading": 90.0, "pitch": 3.0},
    {"id": "street_greek_patriarchate", "name": "Greek Patriarchate Street", "x": 118.0, "y": 185.0, "heading": 90.0, "pitch": 3.0},
    {"id": "street_suq_aftimos", "name": "Suq Aftimos (Muristan Market)", "x": 235.0, "y": 125.0, "heading": 265.0, "pitch": 3.0},
    # Armenian & Jewish Quarter Streets
    {"id": "street_armenian_pat_mid", "name": "Armenian Patriarchate Road Mid-Section", "x": 52.0, "y": -155.0, "heading": 180.0, "pitch": 2.0},
    {"id": "street_ararat", "name": "Ararat Street (Armenian/Jewish Quarter Border)", "x": 195.0, "y": -258.0, "heading": 88.0, "pitch": 3.0},
    {"id": "street_or_hahayim", "name": "Or HaHayim Street", "x": 272.0, "y": -115.0, "heading": 182.0, "pitch": 3.0},
    {"id": "street_misgav_ladach", "name": "Misgav Ladach Street (Jewish Quarter)", "x": 450.0, "y": -115.0, "heading": 173.0, "pitch": 3.0},
    # Randomized multi-angle street checks
    {"id": "rand_muslim_ne_alley", "name": "Random Angle: NE Muslim Quarter Residential Alley", "x": 626.5, "y": 637.5, "heading": 102.0, "pitch": 4.0},
    {"id": "rand_herods_inner_south", "name": "Random Angle: Herod's Gate Inner Street Looking South", "x": 574.0, "y": 625.0, "heading": 192.0, "pitch": 3.0},
]


def main() -> None:
    out_dir = ROOT / "build" / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "poses_landmarks_360.json").write_text(json.dumps(LANDMARKS_360, indent=2), encoding="utf-8")
    (out_dir / "poses_streets_and_exits.json").write_text(json.dumps(STREETS_AND_EXITS, indent=2), encoding="utf-8")
    print(f"Wrote {len(LANDMARKS_360)} landmark 360 poses and {len(STREETS_AND_EXITS)} street/exit poses.")


if __name__ == "__main__":
    main()
