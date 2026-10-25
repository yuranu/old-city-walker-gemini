import * as THREE from 'three';

/**
 * Cartographic Streets-Plan Minimap for the Old City of Jerusalem.
 * Renders a clean vector street plan (`public/data/city_plan.json`) showing:
 *   - 16th-century Ottoman Old City Wall perimeter & 8 named Gates
 *   - Temple Mount (Haram al-Sharif) enclosure & Upper Platform
 *   - Complete OpenStreetMap alley/step network + primary/secondary named streets
 *   - 5 Historic Quarter labels & key monument outlines (zero mesh-AABB blobs!)
 *   - Interactive POI markers + live player position & directional FOV cone
 */
export class Minimap {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas ? canvas.getContext('2d') : null;
    this.places = [];
    this.plan = null;
    this.worldHalfSpan = options.worldHalfSpan ?? 180; // metres around center (scroll to zoom 60m..680m)
    this.centerX = 35;
    this.centerY = 0;
    this.visible = true;
    this.followPlayer = true;
    this.onSelectPlace = options.onSelectPlace || null;
    this.onTeleportXY = options.onTeleportXY || null;

    if (this.canvas) {
      this._bindEvents();
    }
  }

  _bindEvents() {
    this.canvas.addEventListener('click', (e) => {
      e.stopPropagation();
      const rect = this.canvas.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * this.canvas.width;
      const py = ((e.clientY - rect.top) / rect.height) * this.canvas.height;
      const world = this._canvasToENU(px, py);

      // Check if user clicked near a place or gate marker first
      let closestPlace = null;
      let closestDist = Math.max(14, this.worldHalfSpan * 0.075);
      for (const place of this.places) {
        const pxENU = Number(place.x ?? place.camera?.x ?? 0);
        const pyENU = Number(place.y ?? place.camera?.y ?? 0);
        const d = Math.hypot(world.x - pxENU, world.y - pyENU);
        if (d < closestDist) {
          closestDist = d;
          closestPlace = place;
        }
      }

      if (closestPlace && this.onSelectPlace) {
        this.onSelectPlace(closestPlace);
      } else if (this.onTeleportXY) {
        this.onTeleportXY(world.x, world.y);
      }
    });

    this.canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        e.stopPropagation();
        const factor = e.deltaY > 0 ? 1.22 : 0.82;
        this.worldHalfSpan = THREE.MathUtils.clamp(this.worldHalfSpan * factor, 60, 680);
        this.followPlayer = this.worldHalfSpan <= 360;
      },
      { passive: false }
    );
  }

  setPlaces(places) {
    this.places = Array.isArray(places) ? places : [];
  }

  setCityPlan(planData) {
    if (planData && typeof planData === 'object') {
      this.plan = planData;
    }
  }

  toggleFullCityView() {
    if (this.worldHalfSpan > 360) {
      this.worldHalfSpan = 160;
      this.followPlayer = true;
    } else {
      this.worldHalfSpan = 620;
      this.followPlayer = false;
    }
    return this.worldHalfSpan > 360;
  }

  /**
   * Kept for backwards compatibility; no longer extracts scene mesh AABB blobs.
   */
  buildFromScene(_sceneRoot, _manifestData = null) {
    // Intentionally empty: Minimap now renders clean vector streets plan from city_plan.json
  }

  _enuToCanvas(x, y) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const scale = w / (this.worldHalfSpan * 2);
    return {
      cx: w * 0.5 + (x - this.centerX) * scale,
      cy: h * 0.5 - (y - this.centerY) * scale
    };
  }

  _canvasToENU(cx, cy) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const scale = w / (this.worldHalfSpan * 2);
    return {
      x: this.centerX + (cx - w * 0.5) / scale,
      y: this.centerY + (h * 0.5 - cy) / scale
    };
  }

  _tracePolygon(ctx, pts) {
    if (!pts || pts.length < 3) return;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const pt = pts[i];
      const x = Array.isArray(pt) ? pt[0] : pt.x;
      const y = Array.isArray(pt) ? pt[1] : pt.y;
      const { cx, cy } = this._enuToCanvas(x, y);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.closePath();
  }

  _tracePolyline(ctx, pts) {
    if (!pts || pts.length < 2) return false;
    const w = this.canvas.width;
    const h = this.canvas.height;
    let anyVisible = false;
    for (let i = 0; i < pts.length; i++) {
      const { cx, cy } = this._enuToCanvas(pts[i][0], pts[i][1]);
      if (cx >= -40 && cx <= w + 40 && cy >= -40 && cy <= h + 40) {
        anyVisible = true;
        break;
      }
    }
    if (!anyVisible) return false;

    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const { cx, cy } = this._enuToCanvas(pts[i][0], pts[i][1]);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    return true;
  }

  render(playerPose, activePlaceId = null, fovDeg = 65) {
    if (!this.ctx || !this.visible) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const dpr = w / 250; // Scale factor relative to 250px logical size

    if (playerPose) {
      if (this.followPlayer && this.worldHalfSpan <= 360) {
        this.centerX = playerPose.x;
        this.centerY = playerPose.y;
      } else {
        // Center on the heart of the Old City when viewing the full city street plan
        this.centerX = 365;
        this.centerY = 155;
      }
    }

    // 1. Exterior Dark Slate Background
    ctx.fillStyle = '#0e131b';
    ctx.fillRect(0, 0, w, h);

    // 2. Old City Intramural Urban Territory & Temple Mount Esplanade
    if (this.plan?.cityWall) {
      this._tracePolygon(ctx, this.plan.cityWall);
      ctx.fillStyle = '#171f2c';
      ctx.fill();
    }

    if (this.plan?.templeMountWall) {
      this._tracePolygon(ctx, this.plan.templeMountWall);
      ctx.fillStyle = 'rgba(218, 178, 92, 0.09)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(235, 195, 110, 0.42)';
      ctx.lineWidth = 1.2 * dpr;
      ctx.stroke();
    }

    if (this.plan?.domePlatform) {
      this._tracePolygon(ctx, this.plan.domePlatform);
      ctx.fillStyle = 'rgba(235, 195, 110, 0.12)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(243, 200, 120, 0.35)';
      ctx.lineWidth = 0.9 * dpr;
      ctx.stroke();
    }

    // 3. Subtle Cartographic Coordinate Grid
    const gridStep = this.worldHalfSpan > 280 ? 200 : 100;
    ctx.strokeStyle = 'rgba(255, 240, 210, 0.045)';
    ctx.lineWidth = 1 * dpr;
    const startX = Math.floor((this.centerX - this.worldHalfSpan) / gridStep) * gridStep;
    const endX = Math.ceil((this.centerX + this.worldHalfSpan) / gridStep) * gridStep;
    const startY = Math.floor((this.centerY - this.worldHalfSpan) / gridStep) * gridStep;
    const endY = Math.ceil((this.centerY + this.worldHalfSpan) / gridStep) * gridStep;

    for (let gx = startX; gx <= endX; gx += gridStep) {
      const p = this._enuToCanvas(gx, this.centerY);
      ctx.beginPath();
      ctx.moveTo(p.cx, 0);
      ctx.lineTo(p.cx, h);
      ctx.stroke();
    }
    for (let gy = startY; gy <= endY; gy += gridStep) {
      const q = this._enuToCanvas(this.centerX, gy);
      ctx.beginPath();
      ctx.moveTo(0, q.cy);
      ctx.lineTo(w, q.cy);
      ctx.stroke();
    }

    // 4. OpenStreetMap Alley & Stepped Lane Network (Clean Streets Plan)
    const zoomFactor = THREE.MathUtils.clamp(220 / this.worldHalfSpan, 0.45, 2.2);
    if (this.plan?.osmStreets) {
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (const st of this.plan.osmStreets) {
        if (!this._tracePolyline(ctx, st.pts)) continue;
        if (st.tier === 'secondary') {
          ctx.strokeStyle = 'rgba(210, 198, 174, 0.48)';
          ctx.lineWidth = 1.9 * dpr * zoomFactor;
        } else if (st.tier === 'steps') {
          ctx.strokeStyle = 'rgba(195, 178, 148, 0.36)';
          ctx.lineWidth = 1.2 * dpr * zoomFactor;
        } else {
          ctx.strokeStyle = 'rgba(168, 182, 202, 0.30)';
          ctx.lineWidth = 1.1 * dpr * zoomFactor;
        }
        ctx.stroke();
      }
    }

    // 5. Curated Named Primary & Secondary Thoroughfares (Double-Edged Street Casing)
    if (this.plan?.curatedStreets) {
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      // Pass A: Dark street casing
      for (const st of this.plan.curatedStreets) {
        if (!this._tracePolyline(ctx, st.pts)) continue;
        const isPrimary = st.tier === 'primary';
        ctx.strokeStyle = 'rgba(10, 14, 20, 0.85)';
        ctx.lineWidth = (isPrimary ? 4.4 : 3.1) * dpr * Math.max(0.65, zoomFactor * 0.85);
        ctx.stroke();
      }
      // Pass B: Warm parchment/gold street core
      for (const st of this.plan.curatedStreets) {
        if (!this._tracePolyline(ctx, st.pts)) continue;
        const isPrimary = st.tier === 'primary';
        ctx.strokeStyle = isPrimary ? '#e8cf9c' : '#b8c4d4';
        ctx.lineWidth = (isPrimary ? 2.5 : 1.65) * dpr * Math.max(0.65, zoomFactor * 0.85);
        ctx.stroke();
      }
    }

    // 6. Clean Key Monument Outlines (Only the 13 iconic monuments, zero building-mesh blobs)
    if (this.plan?.landmarks) {
      for (const lm of this.plan.landmarks) {
        this._tracePolygon(ctx, lm.poly);
        ctx.fillStyle = 'rgba(243, 200, 120, 0.24)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 224, 156, 0.78)';
        ctx.lineWidth = 1.15 * dpr;
        ctx.stroke();
      }
    }

    // 7. Bold 16th-Century Ottoman Old City Wall Perimeter
    if (this.plan?.cityWall) {
      this._tracePolygon(ctx, this.plan.cityWall);
      ctx.strokeStyle = 'rgba(12, 15, 22, 0.95)';
      ctx.lineWidth = 4.2 * dpr;
      ctx.stroke();
      ctx.strokeStyle = '#eeb962';
      ctx.lineWidth = 2.2 * dpr;
      ctx.stroke();
    }

    // 8. Quarter Labels (when zoomed out or mid-zoom)
    if (this.plan?.quarters && this.worldHalfSpan >= 140) {
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const qFontPx = Math.round((this.worldHalfSpan > 340 ? 8.5 : 9.5) * dpr);
      ctx.font = `700 ${qFontPx}px Inter, sans-serif`;
      for (const q of this.plan.quarters) {
        const { cx, cy } = this._enuToCanvas(q.x, q.y);
        if (cx < 20 || cx > w - 20 || cy < 16 || cy > h - 16) continue;
        ctx.fillStyle = 'rgba(10, 14, 20, 0.78)';
        ctx.fillText(q.name, cx + 1, cy + 1);
        ctx.fillStyle = 'rgba(235, 226, 208, 0.55)';
        ctx.fillText(q.name, cx, cy);
      }
    }

    // 9. Street Name Labels (when zoomed in <= 280m half-span)
    if (this.plan?.curatedStreets && this.worldHalfSpan <= 280) {
      ctx.font = `600 ${Math.round(8.2 * dpr)}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const placedBoxes = [];

      for (const st of this.plan.curatedStreets) {
        if (!st.name || st.pts.length < 2) continue;
        if (this.worldHalfSpan > 175 && st.tier !== 'primary') continue;
        const midIdx = Math.floor(st.pts.length / 2);
        const p0 = st.pts[Math.max(0, midIdx - 1)];
        const p1 = st.pts[midIdx];
        const mx = 0.5 * (p0[0] + p1[0]);
        const my = 0.5 * (p0[1] + p1[1]);
        const { cx, cy } = this._enuToCanvas(mx, my);
        if (cx < 32 || cx > w - 32 || cy < 22 || cy > h - 22) continue;

        const shortLabel = st.name.split('(')[0].split('/')[0].trim();
        const boxW = shortLabel.length * 4.8 * dpr;
        const boxH = 11 * dpr;
        const overlaps = placedBoxes.some(
          (b) => Math.abs(b.x - cx) < (b.w + boxW) * 0.55 && Math.abs(b.y - cy) < (b.h + boxH) * 0.85
        );
        if (overlaps) continue;
        placedBoxes.push({ x: cx, y: cy, w: boxW, h: boxH });

        ctx.lineWidth = 2.6 * dpr;
        ctx.strokeStyle = 'rgba(12, 16, 24, 0.92)';
        ctx.strokeText(shortLabel, cx, cy);
        ctx.fillStyle = st.tier === 'primary' ? '#ffe5ad' : '#d8e2f0';
        ctx.fillText(shortLabel, cx, cy);
      }
    }

    // 10. 8 Named Gate Badges along the City Wall
    if (this.plan?.gates) {
      const showGateNames = this.worldHalfSpan >= 220;
      ctx.font = `700 ${Math.round(7.8 * dpr)}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      for (const g of this.plan.gates) {
        const { cx, cy } = this._enuToCanvas(g.x, g.y);
        if (cx < -10 || cx > w + 10 || cy < -10 || cy > h + 10) continue;
        const sz = 3.6 * dpr;
        ctx.fillStyle = '#141a24';
        ctx.strokeStyle = '#f5c76b';
        ctx.lineWidth = 1.5 * dpr;
        ctx.beginPath();
        ctx.rect(cx - sz, cy - sz, sz * 2, sz * 2);
        ctx.fill();
        ctx.stroke();

        if (showGateNames) {
          const ly = cy - 7.5 * dpr;
          ctx.lineWidth = 2.4 * dpr;
          ctx.strokeStyle = 'rgba(10, 13, 19, 0.92)';
          ctx.strokeText(g.name, cx, ly);
          ctx.fillStyle = '#f7d488';
          ctx.fillText(g.name, cx, ly);
        }
      }
    }

    // 11. Interactive POI Markers
    this.places.forEach((place) => {
      const pxENU = Number(place.x ?? place.camera?.x ?? 0);
      const pyENU = Number(place.y ?? place.camera?.y ?? 0);
      const { cx, cy } = this._enuToCanvas(pxENU, pyENU);
      if (cx < -10 || cx > w + 10 || cy < -10 || cy > h + 10) return;
      const isActive = place.id === activePlaceId;

      ctx.beginPath();
      ctx.arc(cx, cy, (isActive ? 5.5 : 3.6) * dpr, 0, Math.PI * 2);
      ctx.fillStyle = isActive ? '#ffe2a6' : '#243044';
      ctx.fill();
      ctx.lineWidth = (isActive ? 1.8 : 1.2) * dpr;
      ctx.strokeStyle = isActive ? '#ffffff' : '#f3c878';
      ctx.stroke();

      if (isActive) {
        const shortName = place.shortName || place.name.split('—')[0].split('(')[0].trim();
        ctx.font = `700 ${Math.round(8.5 * dpr)}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.lineWidth = 2.8 * dpr;
        ctx.strokeStyle = 'rgba(10, 14, 20, 0.95)';
        ctx.strokeText(shortName, cx, cy - 7 * dpr);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(shortName, cx, cy - 7 * dpr);
      }
    });

    // 12. Player Position + Directional FOV Cone
    if (playerPose) {
      const { cx, cy } = this._enuToCanvas(playerPose.x, playerPose.y);
      const hdgRad = ((playerPose.heading - 90) * Math.PI) / 180;
      const halfFov = ((fovDeg * 0.5) * Math.PI) / 180;
      const coneLen = 30 * dpr;

      const grad = ctx.createRadialGradient(cx, cy, 2 * dpr, cx, cy, coneLen);
      grad.addColorStop(0, 'rgba(74, 212, 255, 0.58)');
      grad.addColorStop(1, 'rgba(74, 212, 255, 0.0)');

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, coneLen, hdgRad - halfFov, hdgRad + halfFov);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, 4.4 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
      ctx.lineWidth = 1.8 * dpr;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }

    // 13. Compass Badge & Scale Bar Footer
    ctx.fillStyle = 'rgba(14, 19, 27, 0.82)';
    ctx.fillRect(0, h - 18 * dpr, w, 18 * dpr);

    ctx.fillStyle = 'rgba(243, 200, 120, 0.92)';
    ctx.font = `700 ${Math.round(9.5 * dpr)}px Inter, sans-serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText('N ▲', w - 7 * dpr, 6 * dpr);

    ctx.fillStyle = 'rgba(226, 218, 202, 0.78)';
    ctx.font = `${Math.round(8.5 * dpr)}px monospace`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const modeLabel = this.worldHalfSpan > 360 ? 'Full Old City Plan' : `Local Streets (±${Math.round(this.worldHalfSpan)}m)`;
    ctx.fillText(`${modeLabel} · Scroll Zoom`, 6 * dpr, h - 9 * dpr);
  }
}
