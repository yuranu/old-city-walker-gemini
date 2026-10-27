#!/usr/bin/env node
/**
 * scripts/roam_capture.mjs
 *
 * Roaming Multi-Angle Screenshot Capture Tool for AI Visual Sanity & Realism Agents.
 * - Captures 1280x800 WebGL screenshots across the entire walkable area from varied
 *   positions, headings (360° sweeps), and pitches (street level, look-up, look-down, overview),
 *   plus side-by-side Reference Photo vs. Game comparison pairs (`pair_<id>.jpg`).
 * - Can also be invoked by an AI Explorer agent with custom poses:
 *     node scripts/roam_capture.mjs --out build/qa/custom --poses '[{"id":"test1","x":25,"y":5,"heading":180,"pitch":0}]'
 */

import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';

const ROOT = process.cwd();

function parseArgs() {
  const args = process.argv.slice(2);
  let outDir = path.join(ROOT, 'build', 'qa', 'sanity1');
  let url = 'http://127.0.0.1:5173/?clean=1';
  let zone = 'all';
  let customPosesJson = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--out' && args[i + 1]) {
      outDir = path.resolve(ROOT, args[++i]);
    } else if (args[i] === '--url' && args[i + 1]) {
      url = args[++i];
    } else if (args[i] === '--zone' && args[i + 1]) {
      zone = args[++i];
    } else if (args[i] === '--poses' && args[i + 1]) {
      customPosesJson = args[++i];
    } else if (args[i] === '--poses-file' && args[i + 1]) {
      customPosesJson = fs.readFileSync(path.resolve(ROOT, args[++i]), 'utf-8');
    }
  }
  return { outDir, url, zone, customPosesJson };
}

export const ROAM_ZONES = {
  zone1_west_and_gate: [
    // West Plaza & Approach from multiple angles
    { id: 'z1_01_west_plaza_east', name: 'West Plaza Looking East at Jaffa Gate', x: -34.0, y: 5.0, z_eye: 1.7, heading: 98, pitch: 4, fov: 70, refId: 'west_plaza' },
    { id: 'z1_02_west_plaza_north_wall', name: 'West Plaza Looking North-East at City Wall', x: -32.0, y: -2.0, z_eye: 1.7, heading: 45, pitch: 8, fov: 75 },
    { id: 'z1_03_west_plaza_south_moat', name: 'West Plaza Looking South-East at Citadel & Wall', x: -28.0, y: 8.0, z_eye: 1.7, heading: 130, pitch: 5, fov: 75 },
    { id: 'z1_04_west_plaza_lookback_west', name: 'West Plaza Looking West toward Mamilla Horizon', x: -20.0, y: 2.0, z_eye: 1.7, heading: 270, pitch: 2, fov: 75 },
    // Jaffa Gate Outer Portal & Bastion
    { id: 'z1_05_gate_portal_front', name: 'Jaffa Gate Outer Portal Straight On', x: -16.0, y: 2.0, z_eye: 1.7, heading: 94, pitch: 8, fov: 70, refId: 'jaffa_gate_portal' },
    { id: 'z1_06_gate_portal_high_lookup', name: 'Jaffa Gate Outer Portal & Machicolation Look-Up', x: -13.5, y: 2.7, z_eye: 1.7, heading: 90, pitch: 28, fov: 75 },
    { id: 'z1_07_gate_portal_oblique_sw', name: 'Jaffa Gate Portal Oblique from South-West', x: -15.5, y: -4.0, z_eye: 1.7, heading: 45, pitch: 12, fov: 75 },
    // Inside Jaffa Gate L-Passage
    { id: 'z1_08_passage_entering_east', name: 'Jaffa Gate Passage Entering East toward Gate Leaves', x: -7.5, y: 2.7, z_eye: 1.7, heading: 90, pitch: 5, fov: 75 },
    { id: 'z1_09_passage_turn_south', name: 'Jaffa Gate Vaulted L-Passage Turning South', x: -1.5, y: 4.5, z_eye: 1.7, heading: 145, pitch: 5, fov: 75, refId: 'jaffa_gate_passage' },
    { id: 'z1_10_passage_vault_ceiling', name: 'Jaffa Gate Passage Interior Vault Ceiling Look-Up', x: 0.5, y: 2.5, z_eye: 1.7, heading: 180, pitch: 40, fov: 80 },
    { id: 'z1_11_passage_lookback_north', name: 'Jaffa Gate South Exit Looking Back North Into Passage', x: 1.0, y: -9.5, z_eye: 1.7, heading: 10, pitch: 14, fov: 75 },
    // 1898 Breach & Moat Roadway
    { id: 'z1_12_breach_1898_se', name: '1898 Wall Breach Looking South-East', x: -15.0, y: -7.5, z_eye: 1.7, heading: 110, pitch: 5, fov: 70, refId: 'breach_1898' },
    { id: 'z1_13_breach_road_looking_west', name: '1898 Breach Road Looking Back West-North-West', x: 10.0, y: -11.0, z_eye: 1.7, heading: 285, pitch: 6, fov: 75 },
    { id: 'z1_14_breach_south_down_moat_rd', name: 'West of Citadel Looking South Along Outer Wall & Moat', x: 6.0, y: -18.0, z_eye: 1.7, heading: 175, pitch: 0, fov: 75 },
  ],

  zone2_citadel_and_omar_sq: [
    // Citadel & Tower of David
    { id: 'z2_01_tower_of_david_nw', name: 'Tower of David (Phasael Tower) & Moat from Breach', x: 5.5, y: -11.5, z_eye: 1.7, heading: 140, pitch: 8, fov: 70, refId: 'tower_of_david' },
    { id: 'z2_02_citadel_moat_lookup_down', name: 'Looking Down Into Sunken Citadel Dry Moat & Glacis', x: 25.0, y: -14.2, z_eye: 1.7, heading: 165, pitch: -18, fov: 75 },
    { id: 'z2_03_citadel_bridge_south', name: 'Citadel Moat Bridge Looking South at Museum Entrance', x: 44.5, y: -14.0, z_eye: 1.7, heading: 185, pitch: 5, fov: 75 },
    { id: 'z2_04_citadel_bridge_west_moat', name: 'From Citadel Moat Bridge Looking West Across Moat & Tower', x: 43.0, y: -21.0, z_eye: 1.7, heading: 265, pitch: 4, fov: 75 },
    // Omar Square 360 Sweep
    { id: 'z2_05_omar_sq_west_looking_east', name: 'Omar Square West Looking East', x: 22.0, y: -2.5, z_eye: 1.7, heading: 85, pitch: 4, fov: 70, refId: 'omar_square_west' },
    { id: 'z2_06_omar_sq_west_looking_north', name: 'Omar Square West Looking North at Tourist Info & Imperial Corner', x: 16.0, y: -2.0, z_eye: 1.7, heading: 15, pitch: 8, fov: 75 },
    { id: 'z2_07_omar_sq_north_imperial', name: 'New Imperial Hotel Facade from Omar Square', x: 26.5, y: -1.5, z_eye: 1.7, heading: 8, pitch: 8, fov: 70, refId: 'omar_square_north' },
    { id: 'z2_08_omar_sq_center_petra', name: 'Omar Square Center Looking NE at Imperial-Petra Junction', x: 50.0, y: -2.0, z_eye: 1.7, heading: 25, pitch: 8, fov: 75 },
    { id: 'z2_09_omar_sq_petra_facade', name: 'Petra Hotel South Facade Looking North', x: 66.0, y: -2.0, z_eye: 1.7, heading: 5, pitch: 10, fov: 75 },
    { id: 'z2_10_omar_sq_east_looking_west', name: 'Omar Square East Looking Back West toward Jaffa Gate', x: 76.0, y: 2.0, z_eye: 1.7, heading: 265, pitch: 5, fov: 70, refId: 'omar_square_east' },
    { id: 'z2_11_omar_sq_east_looking_south', name: 'Omar Square East Looking South toward Armenian Rd & Christian Info', x: 64.0, y: 2.0, z_eye: 1.7, heading: 175, pitch: 2, fov: 75 },
    // Armenian Patriarchate Road (South-East Corridor)
    { id: 'z2_12_armenian_rd_south', name: 'Armenian Patriarchate Road Looking South', x: 60.0, y: -12.0, z_eye: 1.7, heading: 170, pitch: 2, fov: 75 },
    { id: 'z2_13_armenian_rd_east_facade', name: 'Christian Information Centre West/North Facade', x: 58.5, y: -10.0, z_eye: 1.7, heading: 135, pitch: 6, fov: 75 },
    { id: 'z2_14_armenian_rd_lookback_north', name: 'Armenian Patriarchate Road Looking Back North to Omar Sq', x: 59.5, y: -28.0, z_eye: 1.7, heading: 5, pitch: 4, fov: 75 },
  ],

  zone3_latin_and_david_souq: [
    // Latin Patriarchate Road (North Corridor, x = 21.2..26.8, centered at x = 24.0)
    { id: 'z3_01_latin_lane_entry_north', name: 'Latin Patriarchate Road Entry Looking North', x: 24.0, y: 12.0, z_eye: 1.7, heading: 0, pitch: 5, fov: 75 },
    { id: 'z3_02_latin_lane_mid_north', name: 'Latin Patriarchate Road Looking North Through Arch', x: 24.0, y: 22.0, z_eye: 1.7, heading: 0, pitch: 4, fov: 70, refId: 'latin_patriarchate_lane' },
    { id: 'z3_03_latin_lane_west_shops', name: 'Latin Patriarchate Road Looking West at Shopfronts', x: 25.8, y: 34.0, z_eye: 1.7, heading: 285, pitch: 6, fov: 75 },
    { id: 'z3_04_latin_lane_east_shops', name: 'Latin Patriarchate Road Looking East at Shopfronts', x: 22.2, y: 35.0, z_eye: 1.7, heading: 75, pitch: 6, fov: 75 },
    { id: 'z3_05_latin_lane_north_end_south', name: 'Latin Patriarchate Road North End Looking Back South', x: 24.0, y: 52.0, z_eye: 1.7, heading: 180, pitch: 2, fov: 75 },
    // David Street Stepped Souq (East Corridor)
    { id: 'z3_06_david_st_entry_east', name: 'David Street Souq Entrance Looking East Down Steps', x: 78.2, y: 2.0, z_eye: 1.7, heading: 90, pitch: 2, fov: 70, refId: 'david_street_souq' },
    { id: 'z3_07_david_st_entry_north_corner', name: 'David Street Entrance North Corner (Petra to Souq Transition)', x: 76.0, y: -0.5, z_eye: 1.7, heading: 40, pitch: 6, fov: 75 },
    { id: 'z3_08_david_st_entry_south_corner', name: 'David Street Entrance South Corner Transition', x: 76.0, y: 4.5, z_eye: 1.7, heading: 140, pitch: 6, fov: 75 },
    { id: 'z3_09_david_st_mid_east', name: 'David Street Mid Steps Looking East', x: 108.0, y: 2.0, z_eye: 1.7, heading: 90, pitch: 0, fov: 75 },
    { id: 'z3_10_david_st_mid_lookback_west', name: 'David Street Mid Steps Looking Back West Up Toward Omar Sq', x: 112.0, y: 2.0, z_eye: 1.7, heading: 270, pitch: 6, fov: 75 },
    { id: 'z3_11_david_st_lower_east', name: 'David Street Lower Steps Looking East at Terminus', x: 136.0, y: 2.0, z_eye: 1.7, heading: 90, pitch: 2, fov: 75 },
    { id: 'z3_12_david_st_lower_lookback_west', name: 'David Street East Terminus Looking Back West Up Souq', x: 152.0, y: 2.0, z_eye: 1.7, heading: 270, pitch: 8, fov: 75 },
    // Elevated Overview Shots to catch big-picture roof/wall/building collisions
    { id: 'z3_13_overview_omar_sq_from_high', name: 'Elevated Overview of Omar Square & Hotels from Citadel Height', x: 28.0, y: -14.0, z_eye: 14.0, heading: 35, pitch: -12, fov: 80 },
    { id: 'z3_14_overview_gate_and_breach_high', name: 'Elevated Overview of Jaffa Gate, Breach & Citadel Moat', x: 35.0, y: 8.0, z_eye: 14.0, heading: 240, pitch: -12, fov: 80 },
  ],
};

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function fetchJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let body = '';
      res.on('data', (chunk) => (body += chunk));
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (err) {
          reject(err);
        }
      });
    }).on('error', reject);
  });
}

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.nextId = 1;
    this.pending = new Map();
  }
  async connect() {
    this.ws = new WebSocket(this.wsUrl);
    await new Promise((resolve, reject) => {
      this.ws.addEventListener('open', resolve, { once: true });
      this.ws.addEventListener('error', reject, { once: true });
    });
    this.ws.addEventListener('message', (event) => {
      const msg = JSON.parse(typeof event.data === 'string' ? event.data : event.data.toString());
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message || JSON.stringify(msg.error)));
        else resolve(msg.result);
      }
    });
  }
  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async evaluate(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (res.exceptionDetails) {
      throw new Error(`Eval exception: ${res.exceptionDetails.text}`);
    }
    return res.result ? res.result.value : undefined;
  }
  async screenshotJpg() {
    const res = await this.send('Page.captureScreenshot', { format: 'jpeg', quality: 92 });
    return Buffer.from(res.data, 'base64');
  }
  close() {
    if (this.ws) this.ws.close();
  }
}

async function main() {
  const { outDir, url, zone, customPosesJson } = parseArgs();
  fs.mkdirSync(outDir, { recursive: true });

  let poses = [];
  if (customPosesJson) {
    poses = JSON.parse(customPosesJson);
  } else if (zone === 'all') {
    poses = [
      ...ROAM_ZONES.zone1_west_and_gate,
      ...ROAM_ZONES.zone2_citadel_and_omar_sq,
      ...ROAM_ZONES.zone3_latin_and_david_souq,
    ];
  } else if (ROAM_ZONES[zone]) {
    poses = ROAM_ZONES[zone];
  } else {
    throw new Error(`Unknown zone: ${zone}`);
  }

  const debugPort = 9233 + Math.floor(Math.random() * 400);
  const userDataDir = path.join(outDir, `.chrome_profile_${debugPort}`);
  fs.mkdirSync(userDataDir, { recursive: true });

  const chromeArgs = [
    '--headless=new',
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-webgl',
    '--ignore-gpu-blocklist',
    '--window-size=1280,800',
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${userDataDir}`,
    url,
  ];

  const chromeProc = spawn('google-chrome-stable', chromeArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
  let targets = null;
  for (let i = 0; i < 60; i++) {
    try {
      targets = await fetchJson(`http://127.0.0.1:${debugPort}/json/list`);
      if (Array.isArray(targets) && targets.length > 0) break;
    } catch (_) {}
    await sleep(250);
  }
  const pageTarget = targets.find((t) => t.type === 'page') || targets[0];
  const cdp = new CDPClient(pageTarget.webSocketDebuggerUrl);
  await cdp.connect();
  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    width: 1280,
    height: 800,
    deviceScaleFactor: 1,
    mobile: false,
  });

  for (let i = 0; i < 120; i++) {
    const ready = await cdp.evaluate(`Boolean(window.__QA__ && window.__QA__.ready)`);
    if (ready) break;
    await sleep(250);
  }

  const manifest = [];
  for (let i = 0; i < poses.length; i++) {
    const p = poses[i];
    const poseArg = JSON.stringify({
      x: p.x,
      y: p.y,
      z_eye: p.z_eye || 1.7,
      heading: p.heading || 0,
      pitch: p.pitch || 0,
      fov: p.fov || 75,
    });
    const applied = await cdp.evaluate(`window.__QA__.setPose(${poseArg})`);
    const file = `${p.id}.jpg`;
    const buf = await cdp.screenshotJpg();
    fs.writeFileSync(path.join(outDir, file), buf);
    manifest.push({ ...p, file, applied });
    console.log(`[${i + 1}/${poses.length}] Saved ${file} (${p.name})`);
  }

  fs.writeFileSync(path.join(outDir, 'roam_manifest.json'), JSON.stringify(manifest, null, 2));
  cdp.close();
  chromeProc.kill('SIGTERM');
  try {
    fs.rmSync(userDataDir, { recursive: true, force: true });
  } catch (_) {}

  const pyBin = fs.existsSync(path.join(ROOT, '.venv', 'bin', 'python'))
    ? path.join(ROOT, '.venv', 'bin', 'python')
    : 'python3';
  execFileSync(pyBin, [path.join(ROOT, 'scripts', 'make_roam_sheets.py'), outDir], { stdio: 'inherit' });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
