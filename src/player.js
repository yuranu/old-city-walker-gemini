import * as THREE from 'three';

const WALKABLE_EXACT_NAMES = new Set([
  'ground_terrain',
  'street_paving',
  'street_steps',
  'moat_bridge',
  'gate_passage_floor'
]);

const WALKABLE_KEYWORDS = [
  'ground_terrain',
  'street_paving',
  'street_steps',
  'moat_bridge',
  'gate_passage_floor',
  'walkable',
  'paving',
  'pavement',
  'cobble',
  'step',
  'stair',
  'terrain',
  'plaza',
  'road_surface',
  'floor'
];

const NON_WALKABLE_KEYWORDS = [
  'roof',
  'awning',
  'canopy',
  'parapet',
  'battlement',
  'balcony',
  'lintel',
  'arch_top',
  'cornice',
  'dome',
  'minaret',
  'tower',
  'wall',
  'building',
  'facade',
  'shop',
  'shutter',
  'door',
  'sign',
  'tree',
  'foliage',
  'trunk',
  'olive',
  'cypress',
  'pine',
  'palm',
  'planter',
  'bollard',
  'car',
  'bench',
  'lamp',
  'lantern',
  'cable',
  'railing',
  'fence',
  'grille',
  'table',
  'chair',
  'umbrella',
  'parasol',
  'stall',
  'crate',
  'barrel',
  'pot',
  'ac_unit',
  'water_tank'
];

/**
 * Fast 2D XZ Spatial Hash Grid for static mesh triangles.
 * Accelerates ground height queries and horizontal wall bumper rays by ~1000x
 * by testing only triangles inside the player's local 4m x 4m cell instead of full city meshes.
 */
class SpatialTriangleGrid {
  constructor(cellSize = 4.0) {
    this.cellSize = cellSize;
    this.cells = new Map();
  }

  _key(gx, gz) {
    return (gx << 16) ^ (gz & 0xffff);
  }

  buildFromMeshes(meshes, filterHorizontalNormals = false) {
    this.cells.clear();
    const vA = new THREE.Vector3();
    const vB = new THREE.Vector3();
    const vC = new THREE.Vector3();
    const cb = new THREE.Vector3();
    const ab = new THREE.Vector3();
    const cs = this.cellSize;

    for (const mesh of meshes) {
      if (!mesh.isMesh || !mesh.geometry) continue;
      // Skip LOD1/LOD2 duplicates if LOD0 parent or sibling exists
      if (/_lod[12]$/i.test(mesh.name || '')) continue;

      const geo = mesh.geometry;
      const posAttr = geo.attributes?.position;
      if (!posAttr) continue;

      mesh.updateMatrixWorld(true);
      const matWorld = mesh.matrixWorld;
      const indexAttr = geo.index;
      const triCount = indexAttr ? indexAttr.count / 3 : posAttr.count / 3;

      for (let i = 0; i < triCount; i++) {
        let i0 = i * 3;
        let i1 = i * 3 + 1;
        let i2 = i * 3 + 2;
        if (indexAttr) {
          i0 = indexAttr.getX(i0);
          i1 = indexAttr.getX(i1);
          i2 = indexAttr.getX(i2);
        }

        vA.fromBufferAttribute(posAttr, i0).applyMatrix4(matWorld);
        vB.fromBufferAttribute(posAttr, i1).applyMatrix4(matWorld);
        vC.fromBufferAttribute(posAttr, i2).applyMatrix4(matWorld);

        if (filterHorizontalNormals) {
          // Skip high upper-story/dome triangles far above any street-level elevation across the Old City
          if (vA.y > 45.0 && vB.y > 45.0 && vC.y > 45.0) continue;
          cb.subVectors(vC, vB);
          ab.subVectors(vA, vB);
          cb.cross(ab).normalize();
          // For wall bumpers, skip flat horizontal triangles
          if (Math.abs(cb.y) > 0.78) continue;
        }

        const minX = Math.min(vA.x, vB.x, vC.x);
        const maxX = Math.max(vA.x, vB.x, vC.x);
        const minY = Math.min(vA.y, vB.y, vC.y);
        const maxY = Math.max(vA.y, vB.y, vC.y);
        const minZ = Math.min(vA.z, vB.z, vC.z);
        const maxZ = Math.max(vA.z, vB.z, vC.z);

        const tri = {
          a: vA.clone(),
          b: vB.clone(),
          c: vC.clone(),
          minY,
          maxY
        };

        const gx0 = Math.floor(minX / cs);
        const gx1 = Math.floor(maxX / cs);
        const gz0 = Math.floor(minZ / cs);
        const gz1 = Math.floor(maxZ / cs);

        for (let gx = gx0; gx <= gx1; gx++) {
          for (let gz = gz0; gz <= gz1; gz++) {
            const k = this._key(gx, gz);
            let bucket = this.cells.get(k);
            if (!bucket) {
              bucket = [];
              this.cells.set(k, bucket);
            }
            bucket.push(tri);
          }
        }
      }
    }
  }

  getBucketAt(x, z) {
    const gx = Math.floor(x / this.cellSize);
    const gz = Math.floor(z / this.cellSize);
    return this.cells.get(this._key(gx, gz)) || null;
  }
}

/**
 * Classifies scene meshes strictly into walkable ground meshes vs. obstacle meshes.
 * Ensures downward ground raycasts ONLY hit walkable street/step/bridge/terrain surfaces
 * and NEVER snap the player or QA camera onto a building roof or shop awning.
 */
export function classifyCollisionMeshes(rootObject) {
  const walkableMeshes = [];
  const obstacleMeshes = [];

  rootObject.traverse((obj) => {
    if (!obj.isMesh || !obj.geometry) return;
    if (/_lod[12]$/i.test(obj.name || '')) return;

    const meshName = (obj.name || '').toLowerCase();
    const parentName = (obj.parent?.name || '').toLowerCase();
    const matName = (Array.isArray(obj.material)
      ? obj.material.map((m) => m?.name || '').join(' ')
      : obj.material?.name || ''
    ).toLowerCase();
    const combined = `${meshName} ${parentName}`;

    if (
      WALKABLE_EXACT_NAMES.has(meshName) ||
      WALKABLE_EXACT_NAMES.has(parentName) ||
      meshName.startsWith('ground_paving') ||
      meshName.includes('mat_paving') ||
      meshName.includes('paving_flagstone') ||
      obj.userData?.walkable === true
    ) {
      walkableMeshes.push(obj);
      return;
    }

    const hasNonWalkable = NON_WALKABLE_KEYWORDS.some((kw) => combined.includes(kw));
    const hasWalkable =
      WALKABLE_KEYWORDS.some((kw) => combined.includes(kw)) ||
      matName.includes('paving') ||
      matName.includes('flagstone') ||
      matName.includes('asphalt') ||
      matName.includes('ground');

    if (hasWalkable && !hasNonWalkable) {
      walkableMeshes.push(obj);
      return;
    }

    const isNonColliderDetail =
      combined.includes('foliage') ||
      combined.includes('leaf') ||
      combined.includes('leaves') ||
      combined.includes('cable') ||
      combined.includes('wire') ||
      combined.includes('sign') ||
      combined.includes('shopsigns') ||
      combined.includes('buildings_details') ||
      combined.includes('streetprops') ||
      combined.includes('litter') ||
      combined.includes('decal') ||
      matName.includes('foliage');

    if (!isNonColliderDetail) {
      obstacleMeshes.push(obj);
    }
  });

  return { walkableMeshes, obstacleMeshes };
}

export class PlayerController {
  constructor(camera, canvas, options = {}) {
    this.camera = camera;
    this.canvas = canvas;
    this.camera.rotation.order = 'YXZ';

    // Movement & physics constants
    this.eyeHeight = options.eyeHeight ?? 1.7;
    this.walkSpeed = options.walkSpeed ?? 4.2;
    this.sprintSpeed = options.sprintSpeed ?? 7.5;
    this.flySpeed = options.flySpeed ?? 14.0;
    this.jumpVelocity = options.jumpVelocity ?? 5.5;
    this.gravity = options.gravity ?? 16.0;
    this.stepUpMax = options.stepUpMax ?? 0.45;
    this.capsuleRadius = options.capsuleRadius ?? 0.38;

    // State
    this.position = new THREE.Vector3(0, 0, 0);
    this.velocity = new THREE.Vector3(0, 0, 0);
    this.heading = 90;
    this.pitch = 0;
    this.flyMode = false;
    this.isGrounded = true;
    this.isPointerLocked = false;
    this.isDragging = false;
    this.lastDragX = 0;
    this.lastDragY = 0;
    this.isActivelyMoving = false;
    this.lastMouseMoveTime = 0;

    // Collision sets & fast 2D spatial triangle grids
    this.walkableMeshes = [];
    this.obstacleMeshes = [];
    this.walkableGrid = new SpatialTriangleGrid(4.0);
    this.obstacleGrid = new SpatialTriangleGrid(4.0);

    // Input map
    this.keys = new Set();

    // Reusable raycasters with firstHitOnly = true and tight far bounds
    this.downRaycaster = new THREE.Raycaster();
    this.downRaycaster.firstHitOnly = true;
    this.downRaycaster.near = 0.01;
    this.downRaycaster.far = 6.0;

    this.bumperRaycaster = new THREE.Raycaster();
    this.bumperRaycaster.firstHitOnly = true;
    this.bumperRaycaster.near = 0.01;
    this.bumperRaycaster.far = 1.2;

    this._scratchRay = new THREE.Ray();
    this._scratchHit = new THREE.Vector3();

    // 4 cardinal/diagonal horizontal bumper directions at 1 waist height (+0.85m)
    // = 4 horizontal rays + 1 ground ray = 5 ultra-short BVH/grid rays per frame when moving
    this.bumperDirs = [
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(-1, 0, 0),
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0, -1)
    ];

    this.onModeChange = options.onModeChange || null;
    this.onPointerLockChange = options.onPointerLockChange || null;

    this._bindEvents();
    this._syncCamera();
  }

  setCollisionMeshes(walkableMeshes, obstacleMeshes) {
    this.walkableMeshes = walkableMeshes || [];
    this.obstacleMeshes = obstacleMeshes || [];
    this.walkableGrid.buildFromMeshes(this.walkableMeshes, false);
    this.obstacleGrid.buildFromMeshes(this.obstacleMeshes, true);
  }

  _bindEvents() {
    const requestLock = () => {
      if (document.pointerLockElement !== this.canvas && this.canvas.requestPointerLock) {
        try {
          const promise = this.canvas.requestPointerLock();
          if (promise && typeof promise.catch === 'function') {
            promise.catch(() => {});
          }
        } catch (_) {}
      }
    };

    this.canvas.addEventListener('click', requestLock);
    const promptEl = document.getElementById('pointer-prompt');
    if (promptEl) {
      promptEl.addEventListener('click', (e) => {
        e.stopPropagation();
        requestLock();
      });
    }

    document.addEventListener('pointerlockchange', () => {
      this.isPointerLocked = document.pointerLockElement === this.canvas;
      if (this.onPointerLockChange) {
        this.onPointerLockChange(this.isPointerLocked);
      }
    });

    this.canvas.addEventListener('mousedown', (e) => {
      if (!this.isPointerLocked && e.button === 0) {
        this.isDragging = true;
        this.lastDragX = e.clientX;
        this.lastDragY = e.clientY;
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    window.addEventListener('mousemove', (e) => {
      const sensitivity = 0.14;
      if (this.isPointerLocked) {
        const dx = e.movementX || 0;
        const dy = e.movementY || 0;
        if (Math.abs(dx) > 300 || Math.abs(dy) > 300) return;
        this.heading = (this.heading + dx * sensitivity + 360) % 360;
        this.pitch = THREE.MathUtils.clamp(this.pitch - dy * sensitivity, -85, 85);
        this.lastMouseMoveTime = performance.now();
        this._syncCamera();
      } else if (this.isDragging) {
        const dx = e.clientX - this.lastDragX;
        const dy = e.clientY - this.lastDragY;
        this.lastDragX = e.clientX;
        this.lastDragY = e.clientY;
        if (Math.abs(dx) <= 300 && Math.abs(dy) <= 300) {
          this.heading = (this.heading + dx * sensitivity + 360) % 360;
          this.pitch = THREE.MathUtils.clamp(this.pitch - dy * sensitivity, -85, 85);
          this.lastMouseMoveTime = performance.now();
          this._syncCamera();
        }
      }
    });

    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      this.keys.add(e.code);

      if (e.code === 'KeyF') {
        this.toggleFlyMode();
      } else if (e.code === 'Space' && !this.flyMode && this.isGrounded) {
        this.velocity.y = this.jumpVelocity;
        this.isGrounded = false;
      }
    });

    window.addEventListener('keyup', (e) => {
      this.keys.delete(e.code);
    });

    window.addEventListener('blur', () => {
      this.keys.clear();
      this.isDragging = false;
    });
  }

  toggleFlyMode(forceState) {
    this.flyMode = typeof forceState === 'boolean' ? forceState : !this.flyMode;
    this.velocity.set(0, 0, 0);
    if (!this.flyMode) {
      const groundY = this.sampleGroundHeight(this.position.x, this.position.z, null);
      if (groundY !== null) {
        this.position.y = groundY;
        this.isGrounded = true;
      }
    }
    if (this.onModeChange) {
      this.onModeChange(this.flyMode);
    }
    this._syncCamera();
  }

  /**
   * Samples walkable ground height at Three.js world (threeX, threeZ).
   * Uses the fast 4m x 4m spatial triangle grid first (with far = 6.0 during walking),
   * falling back to Three.js raycaster only if needed.
   */
  sampleGroundHeight(threeX, threeZ, referenceFootY = null) {
    if (!this.walkableMeshes || this.walkableMeshes.length === 0) {
      return 0;
    }

    const rayOriginY = referenceFootY !== null ? referenceFootY + this.stepUpMax + 1.2 : 220;
    const maxRayDist = referenceFootY !== null ? 6.0 : 500.0;

    // 1. Fast spatial grid lookup (~10-25 triangles tested instead of full scene)
    const bucket = this.walkableGrid.getBucketAt(threeX, threeZ);
    if (bucket && bucket.length > 0) {
      this._scratchRay.origin.set(threeX, rayOriginY, threeZ);
      this._scratchRay.direction.set(0, -1, 0);

      let bestY = null;
      const minAllowedY = rayOriginY - maxRayDist;
      const maxAllowedY = referenceFootY !== null ? referenceFootY + this.stepUpMax + 0.15 : Infinity;

      for (let i = 0; i < bucket.length; i++) {
        const tri = bucket[i];
        if (tri.maxY < minAllowedY || tri.minY > rayOriginY) continue;
        const hitPt = this._scratchRay.intersectTriangle(tri.a, tri.b, tri.c, false, this._scratchHit);
        if (hitPt) {
          const y = hitPt.y;
          if (y >= minAllowedY && y <= maxAllowedY) {
            if (bestY === null || y > bestY) {
              bestY = y;
            }
          }
        }
      }

      if (bestY !== null) {
        return bestY;
      }
    }

    // 2. Fallback to standard Raycaster with tight far bound
    this.downRaycaster.near = 0.01;
    this.downRaycaster.far = maxRayDist;
    this.downRaycaster.set(
      new THREE.Vector3(threeX, rayOriginY, threeZ),
      new THREE.Vector3(0, -1, 0)
    );

    const hits = this.downRaycaster.intersectObjects(this.walkableMeshes, false);
    if (hits.length === 0 && referenceFootY !== null) {
      this.downRaycaster.far = 500.0;
      this.downRaycaster.set(
        new THREE.Vector3(threeX, 220, threeZ),
        new THREE.Vector3(0, -1, 0)
      );
      const fallbackHits = this.downRaycaster.intersectObjects(this.walkableMeshes, false);
      if (fallbackHits.length === 0) return null;
      return this._pickBestWalkableHit(fallbackHits, null);
    }

    if (hits.length === 0) {
      if (referenceFootY === null) {
        const offsets = [
          [1.8, 0],
          [-1.8, 0],
          [0, 1.8],
          [0, -1.8],
          [3.2, 0],
          [-3.2, 0]
        ];
        for (const [ox, oz] of offsets) {
          this.downRaycaster.set(
            new THREE.Vector3(threeX + ox, rayOriginY, threeZ + oz),
            new THREE.Vector3(0, -1, 0)
          );
          const nearHits = this.downRaycaster.intersectObjects(this.walkableMeshes, false);
          if (nearHits.length > 0) {
            return this._pickBestWalkableHit(nearHits, null);
          }
        }
      }
      return null;
    }
    return this._pickBestWalkableHit(hits, referenceFootY);
  }

  _pickBestWalkableHit(hits, referenceFootY) {
    let bestY = null;
    for (const hit of hits) {
      const y = hit.point.y;
      if (referenceFootY !== null && y > referenceFootY + this.stepUpMax + 0.15) {
        continue;
      }
      if (bestY === null || y > bestY) {
        bestY = y;
      }
    }
    if (bestY !== null) return bestY;
    return hits[0].point.y;
  }

  setPoseENU({ x = 0, y = 0, z_eye = 1.7, heading = 90, pitch = 0, fov = null }) {
    const threeX = Number(x) || 0;
    const threeZ = -(Number(y) || 0);
    const eyeOffset = Number.isFinite(Number(z_eye)) ? Number(z_eye) : this.eyeHeight;

    this.eyeHeight = eyeOffset;
    this.heading = ((Number(heading) || 0) % 360 + 360) % 360;
    this.pitch = THREE.MathUtils.clamp(Number(pitch) || 0, -85, 85);

    if (fov !== null && Number.isFinite(Number(fov))) {
      this.camera.fov = Number(fov);
      this.camera.updateProjectionMatrix();
    }

    const groundZ = this.sampleGroundHeight(threeX, threeZ, null) ?? 0;
    this.position.set(threeX, groundZ, threeZ);
    this.velocity.set(0, 0, 0);
    this.isGrounded = true;
    this._syncCamera();

    return {
      x: threeX,
      y: -threeZ,
      groundZ,
      eyeY: groundZ + this.eyeHeight,
      heading: this.heading,
      pitch: this.pitch
    };
  }

  getPoseENU() {
    return {
      x: this.position.x,
      y: -this.position.z,
      groundZ: this.position.y,
      eyeY: this.camera.position.y,
      heading: this.heading,
      pitch: this.pitch
    };
  }

  _syncCamera() {
    this.camera.position.set(
      this.position.x,
      this.position.y + this.eyeHeight,
      this.position.z
    );
    const yawRad = -THREE.MathUtils.degToRad(this.heading);
    const pitchRad = THREE.MathUtils.degToRad(this.pitch);
    this.camera.rotation.set(pitchRad, yawRad, 0, 'YXZ');
    this.camera.updateMatrixWorld(true);
  }

  _resolveHorizontalCollisions(candidatePos) {
    if (!this.obstacleMeshes || this.obstacleMeshes.length === 0) return;

    // 4 cardinal rays at 1 waist height (+0.85m) with far = 1.2m
    const originY = candidatePos.y + 0.85;
    const radius = this.capsuleRadius;
    const maxRayDist = 1.2;

    const bucket = this.obstacleGrid.getBucketAt(candidatePos.x, candidatePos.z);
    if (bucket && bucket.length > 0) {
      for (let d = 0; d < 4; d++) {
        const dir = this.bumperDirs[d];
        this._scratchRay.origin.set(candidatePos.x, originY, candidatePos.z);
        this._scratchRay.direction.copy(dir);

        let closestDist = maxRayDist;
        for (let i = 0; i < bucket.length; i++) {
          const tri = bucket[i];
          if (originY < tri.minY - 0.1 || originY > tri.maxY + 0.1) continue;
          const hitPt = this._scratchRay.intersectTriangle(tri.a, tri.b, tri.c, false, this._scratchHit);
          if (hitPt) {
            const dist = this._scratchRay.origin.distanceTo(hitPt);
            if (dist < closestDist) {
              closestDist = dist;
            }
          }
        }

        if (closestDist < radius) {
          const pushDist = radius - closestDist;
          candidatePos.x -= dir.x * pushDist;
          candidatePos.z -= dir.z * pushDist;
        }
      }
      return;
    }

    // Fallback if outside grid cell
    this.bumperRaycaster.far = maxRayDist;
    for (let d = 0; d < 4; d++) {
      const dir = this.bumperDirs[d];
      this.bumperRaycaster.set(
        new THREE.Vector3(candidatePos.x, originY, candidatePos.z),
        dir
      );
      const hits = this.bumperRaycaster.intersectObjects(this.obstacleMeshes, false);
      if (hits.length > 0 && hits[0].distance < radius) {
        const pushDist = radius - hits[0].distance;
        candidatePos.x -= dir.x * pushDist;
        candidatePos.z -= dir.z * pushDist;
      }
    }
  }

  update(dt) {
    const clampedDt = Math.min(dt, 0.1);

    const lookYawSpeed = 95;
    const lookPitchSpeed = 75;
    let lookChanged = false;

    if (this.keys.has('ArrowLeft') || this.keys.has('KeyQ')) {
      this.heading = (this.heading - lookYawSpeed * clampedDt + 360) % 360;
      lookChanged = true;
    }
    if (this.keys.has('ArrowRight') || this.keys.has('KeyE')) {
      this.heading = (this.heading + lookYawSpeed * clampedDt + 360) % 360;
      lookChanged = true;
    }
    if (this.keys.has('ArrowUp')) {
      this.pitch = THREE.MathUtils.clamp(this.pitch + lookPitchSpeed * clampedDt, -85, 85);
      lookChanged = true;
    }
    if (this.keys.has('ArrowDown')) {
      this.pitch = THREE.MathUtils.clamp(this.pitch - lookPitchSpeed * clampedDt, -85, 85);
      lookChanged = true;
    }

    const hdgRad = THREE.MathUtils.degToRad(this.heading);
    const forwardX = Math.sin(hdgRad);
    const forwardZ = -Math.cos(hdgRad);
    const rightX = Math.cos(hdgRad);
    const rightZ = Math.sin(hdgRad);

    let moveForward = 0;
    let moveRight = 0;
    if (this.keys.has('KeyW')) moveForward += 1;
    if (this.keys.has('KeyS')) moveForward -= 1;
    if (this.keys.has('KeyD')) moveRight += 1;
    if (this.keys.has('KeyA')) moveRight -= 1;

    const isSprinting = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');

    if (this.flyMode) {
      const speed = isSprinting ? this.flySpeed * 1.8 : this.flySpeed;
      const pitchRad = THREE.MathUtils.degToRad(this.pitch);
      const cosPitch = Math.cos(pitchRad);
      const sinPitch = Math.sin(pitchRad);

      const dir = new THREE.Vector3(
        forwardX * cosPitch * moveForward + rightX * moveRight,
        sinPitch * moveForward + (this.keys.has('Space') ? 0.8 : 0) - (this.keys.has('ControlLeft') ? 0.8 : 0),
        forwardZ * cosPitch * moveForward + rightZ * moveRight
      );
      if (dir.lengthSq() > 1) dir.normalize();

      this.isActivelyMoving = dir.lengthSq() > 0.001 || lookChanged || (performance.now() - this.lastMouseMoveTime < 120);
      if (dir.lengthSq() > 0.0001 || lookChanged) {
        this.position.addScaledVector(dir, speed * clampedDt);
        this._syncCamera();
      }
      return;
    }

    const speed = isSprinting ? this.sprintSpeed : this.walkSpeed;
    let moveX = forwardX * moveForward + rightX * moveRight;
    let moveZ = forwardZ * moveForward + rightZ * moveRight;
    const len = Math.hypot(moveX, moveZ);

    this.isActivelyMoving =
      len > 0.0001 ||
      !this.isGrounded ||
      lookChanged ||
      performance.now() - this.lastMouseMoveTime < 120;

    // Fast path: if stationary on the ground, skip all raycasting!
    if (len <= 0.0001 && this.isGrounded) {
      if (lookChanged) {
        this._syncCamera();
      }
      return;
    }

    if (len > 0.0001) {
      moveX = (moveX / len) * speed;
      moveZ = (moveZ / len) * speed;
    }

    const candidatePos = this.position.clone();
    candidatePos.x += moveX * clampedDt;
    candidatePos.z += moveZ * clampedDt;

    // Only run 4-ray waist horizontal collision when translating in XZ
    if (len > 0.0001) {
      this._resolveHorizontalCollisions(candidatePos);
    }

    if (!this.isGrounded) {
      this.velocity.y -= this.gravity * clampedDt;
      candidatePos.y += this.velocity.y * clampedDt;
    }

    // 1 ground ray with far = 6.0
    const groundY = this.sampleGroundHeight(
      candidatePos.x,
      candidatePos.z,
      this.isGrounded ? this.position.y : candidatePos.y
    );

    if (groundY !== null) {
      if (this.isGrounded) {
        const deltaY = groundY - this.position.y;
        if (deltaY <= this.stepUpMax) {
          candidatePos.y = THREE.MathUtils.lerp(this.position.y, groundY, Math.min(1, clampedDt * 22));
          if (Math.abs(groundY - candidatePos.y) < 0.02) {
            candidatePos.y = groundY;
          }
        } else {
          candidatePos.x = this.position.x;
          candidatePos.z = this.position.z;
        }
      } else if (candidatePos.y <= groundY && this.velocity.y <= 0) {
        candidatePos.y = groundY;
        this.velocity.y = 0;
        this.isGrounded = true;
      }
    }

    // Full Old City of Jerusalem bounds (West Plaza x=-220 to Eastern Wall x=+950, South Wall y=-520 to North Wall y=+760)
    candidatePos.x = THREE.MathUtils.clamp(candidatePos.x, -220, 950);
    candidatePos.z = THREE.MathUtils.clamp(candidatePos.z, -760, 520);

    this.position.copy(candidatePos);
    this._syncCamera();
  }
}
