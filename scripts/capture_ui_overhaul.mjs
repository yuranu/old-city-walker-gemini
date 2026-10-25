#!/usr/bin/env node
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';

const ROOT = process.cwd();
const outDir = path.join(ROOT, 'build', 'qa', 'ui_overhaul');
fs.mkdirSync(outDir, { recursive: true });

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
function fetchJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let body = '';
      res.on('data', (c) => (body += c));
      res.on('end', () => {
        try { resolve(JSON.parse(body)); } catch (e) { reject(e); }
      });
    }).on('error', reject);
  });
}

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
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
        if (msg.error) reject(new Error(msg.error.message));
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
    const res = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    return res.result ? res.result.value : undefined;
  }
  async screenshotJpg() {
    const res = await this.send('Page.captureScreenshot', { format: 'jpeg', quality: 92 });
    return Buffer.from(res.data, 'base64');
  }
  close() { if (this.ws) this.ws.close(); }
}

async function main() {
  const debugPort = 9688;
  const userDataDir = path.join(outDir, `.chrome_profile_${debugPort}`);
  fs.mkdirSync(userDataDir, { recursive: true });
  const chromeProc = spawn('google-chrome-stable', [
    '--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
    '--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl',
    '--ignore-gpu-blocklist', '--window-size=1280,800',
    `--remote-debugging-port=${debugPort}`, `--user-data-dir=${userDataDir}`,
    'http://127.0.0.1:5173/'
  ], { stdio: 'ignore' });

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
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 800, deviceScaleFactor: 1, mobile: false });

  for (let i = 0; i < 120; i++) {
    const ready = await cdp.evaluate(`Boolean(window.__QA__ && window.__QA__.ready)`);
    if (ready) break;
    await sleep(250);
  }

  // 1. Capture HUD with crisp Streets-Plan Minimap & compact bottom Teleport pill
  await cdp.evaluate(`
    document.getElementById('fps-prompt')?.classList.add('hidden');
    window.__QA__.setPose({ x: 258, y: 122, heading: 5, pitch: 10 });
  `);
  await sleep(300);
  fs.writeFileSync(path.join(outDir, 'ui_minimap_and_hud.jpg'), await cdp.screenshotJpg());

  // 2. Click Teleport button to open <dialog id="teleport-dialog" closedby="any">
  await cdp.evaluate(`document.getElementById('btn-open-teleport')?.click()`);
  await sleep(400);
  fs.writeFileSync(path.join(outDir, 'ui_teleport_modal.jpg'), await cdp.screenshotJpg());

  cdp.close();
  chromeProc.kill('SIGTERM');
  try { fs.rmSync(userDataDir, { recursive: true, force: true }); } catch (_) {}
  console.log('Saved ui_minimap_and_hud.jpg and ui_teleport_modal.jpg');
}

main().catch((e) => { console.error(e); process.exit(1); });
