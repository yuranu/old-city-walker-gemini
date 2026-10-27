import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { PlayerController, classifyCollisionMeshes } from './player.js';
import { Minimap } from './minimap.js';

// Built-in fallback list of the 10 recognizable places around Jaffa Gate
const FALLBACK_PLACES = [
  {
    id: 'west_plaza',
    name: 'Jaffa Gate West Approach Plaza',
    subtitle: '16th-century Ottoman gatehouse & machicolations from the western plaza',
    x: -34.0,
    y: 5.0,
    z_eye: 1.7,
    heading: 98.0,
    pitch: 4.0,
    fov: 70.0,
    photo: '/references/west_plaza.jpg'
  },
  {
    id: 'jaffa_gate_portal',
    name: 'Jaffa Gate (Bab el-Khalil) Outer Portal',
    subtitle: 'Monumental pointed arch, Arabic foundation inscription, and iron-plated doors',
    x: -16.0,
    y: 2.0,
    z_eye: 1.7,
    heading: 94.0,
    pitch: 8.0,
    fov: 70.0,
    photo: '/references/jaffa_gate_portal.jpg'
  },
  {
    id: 'jaffa_gate_passage',
    name: 'Jaffa Gate Vaulted L-Passage',
    subtitle: 'Vaulted Ottoman defensive entryway with historic stone groin vaulting',
    x: -1.5,
    y: 4.5,
    z_eye: 1.7,
    heading: 145.0,
    pitch: 5.0,
    fov: 75.0,
    photo: '/references/jaffa_gate_passage.jpg'
  },
  {
    id: 'breach_1898',
    name: '1898 Wall Breach & Moat Road',
    subtitle: 'Historic carriageway opening between Jaffa Gate and the Citadel moat',
    x: -15.0,
    y: -7.5,
    z_eye: 1.7,
    heading: 110.0,
    pitch: 5.0,
    fov: 70.0,
    photo: '/references/breach_1898.jpg'
  },
  {
    id: 'tower_of_david',
    name: 'Tower of David (Phasael Tower) & Citadel Moat',
    subtitle: 'Herodian stone bossing, Mamluk/Ottoman glacis, and moat bridge entrance',
    x: 5.5,
    y: -11.5,
    z_eye: 1.7,
    heading: 140.0,
    pitch: 8.0,
    fov: 70.0,
    photo: '/references/tower_of_david.jpg'
  },
  {
    id: 'omar_square_west',
    name: 'Omar Ibn El-Khattab Square (Looking East)',
    subtitle: 'Bustling inner plaza inside Jaffa Gate looking east toward the Old City',
    x: 22.0,
    y: -2.5,
    z_eye: 1.7,
    heading: 85.0,
    pitch: 4.0,
    fov: 70.0,
    photo: '/references/omar_square_west.jpg'
  },
  {
    id: 'omar_square_north',
    name: 'New Imperial Hotel & Petra Hotel North Facades',
    subtitle: '19th-century stone arcade, iron balconies, and ground-floor bazaar shops',
    x: 26.5,
    y: -1.5,
    z_eye: 1.7,
    heading: 8.0,
    pitch: 8.0,
    fov: 70.0,
    photo: '/references/omar_square_north.jpg'
  },
  {
    id: 'omar_square_east',
    name: 'Omar Square East (Looking Back to Jaffa Gate)',
    subtitle: 'View west across Omar Ibn El-Khattab Square toward the Citadel and gate',
    x: 76.0,
    y: 2.0,
    z_eye: 1.7,
    heading: 265.0,
    pitch: 5.0,
    fov: 70.0,
    photo: '/references/omar_square_east.jpg'
  },
  {
    id: 'latin_patriarchate_lane',
    name: 'Latin Patriarchate Road',
    subtitle: 'Walled stone lane heading north into the Christian Quarter',
    x: 24.0,
    y: 22.0,
    z_eye: 1.7,
    heading: 0.0,
    pitch: 4.0,
    fov: 70.0,
    photo: '/references/latin_patriarchate_lane.jpg'
  },
  {
    id: 'david_street_souq',
    name: 'David Street Stepped Souq Entrance',
    subtitle: 'Narrow stepped market descent lined with colorful Old City bazaar stalls',
    x: 78.2,
    y: 2.0,
    z_eye: 1.7,
    heading: 90.0,
    pitch: 2.0,
    fov: 70.0,
    photo: '/references/david_street_souq.jpg'
  }
];

function normalizePlace(raw, idx = 0) {
  const cam = raw.camera || raw.pose || raw.viewpoint || {};
  const id = raw.id || raw.place_id || `place_${idx + 1}`;
  const x = Number(raw.x ?? cam.x ?? raw.enu_x ?? 0);
  const y = Number(raw.y ?? cam.y ?? raw.enu_y ?? 0);
  const z_eye = Number(raw.z_eye ?? cam.z_eye ?? raw.eye_z ?? cam.eye_z ?? 1.7);
  const heading = Number(raw.heading ?? cam.heading ?? raw.heading_deg ?? cam.heading_deg ?? 90);
  const pitch = Number(raw.pitch ?? cam.pitch ?? raw.pitch_deg ?? cam.pitch_deg ?? 0);
  const fov = Number(raw.fov ?? cam.fov ?? raw.fov_deg ?? cam.fov_deg ?? 65);

  let photo = raw.photo || raw.reference_photo || raw.reference_image || raw.image || `/references/${id}.jpg`;
  if (!photo.startsWith('/') && !photo.startsWith('http')) {
    photo = `/references/${photo.replace(/^references\//, '')}`;
  }

  return {
    ...raw,
    id,
    name: raw.name || raw.title || id,
    subtitle: raw.subtitle || raw.description || raw.notes || 'Historic landmark in the Jaffa Gate district',
    x,
    y,
    z_eye,
    heading,
    pitch,
    fov,
    photo
  };
}

function createSkyDome(sunDirection) {
  const skyGeo = new THREE.SphereGeometry(1800, 36, 18);
  const skyUniforms = {
    uSunDir: { value: sunDirection.clone().normalize() },
    uSkyKey: { value: 0.0 }
  };

  const skyMat = new THREE.ShaderMaterial({
    uniforms: skyUniforms,
    side: THREE.BackSide,
    depthWrite: false,
    vertexShader: /* glsl */ `
      varying vec3 vWorldPosition;
      void main() {
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPos.xyz;
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSunDir;
      uniform float uSkyKey;
      varying vec3 vWorldPosition;

      void main() {
        vec3 dir = normalize(vWorldPosition - cameraPosition);

        // Automated QA Ground-Hole Detection Key:
        // When uSkyKey > 0.5, render pure bright magenta below the horizon
        if (uSkyKey > 0.5 && dir.y < 0.015) {
          gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
          return;
        }

        float elevation = clamp(dir.y, -0.25, 1.0);
        float horizonFactor = pow(1.0 - clamp(elevation, 0.0, 1.0), 3.2);

        // Warm Mediterranean afternoon Rayleigh sky gradient
        vec3 zenithColor = vec3(0.22, 0.46, 0.80);
        vec3 midSkyColor = vec3(0.52, 0.72, 0.92);
        vec3 horizonColor = vec3(0.95, 0.87, 0.74);
        vec3 groundBounce = vec3(0.62, 0.53, 0.41);

        vec3 sky = mix(midSkyColor, zenithColor, smoothstep(0.08, 0.78, max(elevation, 0.0)));
        sky = mix(sky, horizonColor, horizonFactor);

        if (elevation < 0.0) {
          sky = mix(horizonColor, groundBounce, clamp(-elevation * 4.5, 0.0, 1.0));
        }

        // Mie forward scattering around the afternoon sun
        float sunCos = max(dot(dir, normalize(uSunDir)), 0.0);
        float mieHalo = pow(sunCos, 10.0) * 0.38 + pow(sunCos, 96.0) * 0.80;
        float sunDisk = smoothstep(0.9993, 0.99985, sunCos) * 2.6;
        sky += vec3(1.0, 0.90, 0.70) * (mieHalo + sunDisk);

        gl_FragColor = vec4(sky, 1.0);
      }
    `
  });

  const skyMesh = new THREE.Mesh(skyGeo, skyMat);
  skyMesh.name = 'sky_dome';
  skyMesh.frustumCulled = false;
  return { skyMesh, skyUniforms, skyMat };
}

/**
 * Generates a real-time Sky & Limestone Bounce IBL Environment Map via PMREMGenerator
 * so glazed windows, worn limestone setts, and metal signs have subtle realistic specular response.
 */
function buildSkyIBL(renderer, sunDirection) {
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  pmremGenerator.compileCubemapShader();

  const envScene = new THREE.Scene();
  const { skyMesh } = createSkyDome(sunDirection);
  skyMesh.scale.setScalar(0.05);
  envScene.add(skyMesh);

  const probeSun = new THREE.DirectionalLight(0xffeed2, 3.5);
  probeSun.position.copy(sunDirection).multiplyScalar(50);
  envScene.add(probeSun);

  const probeHemi = new THREE.HemisphereLight(0xd8e8f8, 0x8c7a62, 1.0);
  envScene.add(probeHemi);

  const envRT = pmremGenerator.fromScene(envScene, 0.02);
  pmremGenerator.dispose();
  return envRT.texture;
}

/**
 * Shared External PBR Texture Library (`/textures/*`).
 * Allows per-region `.glb` files (`public/models/regions/r01..r08.glb`) to be exported
 * as lightweight geometry-only chunks (~2–5 MB each) without re-embedding 17.5 MB of
 * identical textures in every region file, enabling fast delta uploads to Cloudflare!
 */
const SHARED_PBR_SPECS = {
  mat_stone_ashlar: { prefix: 'stone_ashlar', normalStrength: 1.2, metalness: 0.0 },
  mat_stone_ashlar_aged: { prefix: 'stone_ashlar_aged', normalStrength: 1.2, metalness: 0.0 },
  mat_stone_mizzi: { prefix: 'stone_mizzi', normalStrength: 1.1, metalness: 0.0 },
  mat_stone_boss: { prefix: 'stone_boss', normalStrength: 1.25, metalness: 0.0 },
  mat_stone_rubble: { prefix: 'stone_rubble', normalStrength: 1.25, metalness: 0.0 },
  mat_paving: { prefix: 'paving_setts', normalStrength: 1.15, metalness: 0.0 },
  mat_paving_flagstone: { prefix: 'paving_flagstone', normalStrength: 1.05, metalness: 0.0 },
  mat_steel_shutters: { prefix: 'steel_shutters', normalStrength: 1.2, metalness: 0.35 },
  mat_plaster: { prefix: 'plaster_warm', normalStrength: 1.1, metalness: 0.0 },
  mat_roof_tile: { prefix: 'roof_tile', normalStrength: 1.2, metalness: 0.0 },
  mat_wood_doors: { prefix: 'wood_doors', normalStrength: 1.2, metalness: 0.0 },
  mat_bark: { prefix: 'bark', normalStrength: 1.4, metalness: 0.0 },
  mat_iron_dark: { prefix: 'metal_iron', normalStrength: 0.95, metalness: 0.72 },
  mat_trash_black: { prefix: 'trash_bag', normalStrength: 1.1, metalness: 0.02 },
  mat_trash_blue: { prefix: 'trash_blue', normalStrength: 1.1, metalness: 0.02 },
  mat_cardboard: { prefix: 'cardboard', normalStrength: 0.95, metalness: 0.0 },
  mat_terracotta: { prefix: 'terracotta', normalStrength: 0.95, metalness: 0.0 },
  mat_awning_red: { prefix: 'awning_red', normalStrength: 0.9, metalness: 0.0 },
  mat_awning_green: { prefix: 'awning_green', normalStrength: 0.9, metalness: 0.0 },
  mat_awning_striped: { prefix: 'awning_striped', normalStrength: 0.9, metalness: 0.0 },
  mat_corrugated_metal: { prefix: 'corrugated_metal', normalStrength: 1.15, metalness: 0.55 },
  mat_lantern_glow: {
    prefix: 'plaster_warm',
    normalStrength: 0.2,
    metalness: 0.0,
    emissive: 0xfae06b,
    emissiveIntensity: 2.8
  },
  mat_dome_gold: { prefix: 'dome_gold', normalStrength: 1.1, metalness: 0.78 },
  mat_dome_lead: { prefix: 'dome_lead', normalStrength: 1.15, metalness: 0.48 },
  mat_tile_iznik: { prefix: 'tile_iznik', normalStrength: 1.05, metalness: 0.08 }
};

const SHARED_CUTOUT_SPECS = {
  mat_foliage_olive: { file: 'foliage_olive.png', roughness: 0.68, alphaTest: 0.35 },
  mat_foliage_cypress: { file: 'foliage_cypress.png', roughness: 0.70, alphaTest: 0.35 },
  mat_iron_grille: { file: 'iron_grille.png', roughness: 0.42, alphaTest: 0.35 },
  mat_signs: { file: 'sign_atlas.png', roughness: 0.45, alphaTest: 0.0 }
};

async function preloadSharedTextures(maxAniso) {
  const texLoader = new THREE.TextureLoader();
  const targetAniso = Math.min(maxAniso, 8);
  const cache = new Map();

  const loadTex = (url, isColor, repeat = true) => {
    if (cache.has(url)) return cache.get(url);
    const p = new Promise((resolve) => {
      texLoader.load(
        url,
        (tex) => {
          tex.flipY = false; // Match glTF 2.0 TEXCOORD_0 upper-left UV origin
          tex.colorSpace = isColor ? THREE.SRGBColorSpace : THREE.NoColorSpace;
          tex.wrapS = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
          tex.wrapT = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
          tex.anisotropy = targetAniso;
          tex.needsUpdate = true;
          resolve(tex);
        },
        undefined,
        () => resolve(null)
      );
    });
    cache.set(url, p);
    return p;
  };

  const pbrSets = {};
  const tasks = [];

  for (const [matKey, spec] of Object.entries(SHARED_PBR_SPECS)) {
    tasks.push(
      Promise.all([
        loadTex(`/textures/${spec.prefix}_diff.jpg`, true, true),
        loadTex(`/textures/${spec.prefix}_norm.jpg`, false, true),
        loadTex(`/textures/${spec.prefix}_rough.jpg`, false, true)
      ]).then(([map, normalMap, roughnessMap]) => {
        pbrSets[matKey] = { ...spec, map, normalMap, roughnessMap };
      })
    );
  }

  for (const [matKey, spec] of Object.entries(SHARED_CUTOUT_SPECS)) {
    const isRepeat = matKey !== 'mat_signs';
    tasks.push(
      loadTex(`/textures/${spec.file}`, true, isRepeat).then((map) => {
        pbrSets[matKey] = { ...spec, map, normalMap: null, roughnessMap: null };
      })
    );
  }

  await Promise.all(tasks);
  return pbrSets;
}

function enhanceMaterial(mat, child, maxAniso, sharedTextures = null) {
  if (!mat) return;

  const rawName = (mat.name || '').replace(/\.\d+$/, '');
  const matName = rawName.toLowerCase();
  const meshName = `${child.name || ''} ${child.parent?.name || ''}`.toLowerCase();

  // Bind external shared PBR textures if this region .glb was exported as geometry-only
  if (sharedTextures && !mat.map) {
    const shared = sharedTextures[rawName] || sharedTextures[matName];
    if (shared) {
      if (shared.map) {
        mat.map = shared.map;
        mat.color.setRGB(1.0, 1.0, 1.0);
      }
      if (shared.normalMap) {
        mat.normalMap = shared.normalMap;
      }
      if (shared.roughnessMap) {
        mat.roughnessMap = shared.roughnessMap;
        mat.roughness = 1.0;
      } else if (typeof shared.roughness === 'number') {
        mat.roughness = shared.roughness;
      }
      if (typeof shared.metalness === 'number') {
        mat.metalness = shared.metalness;
      }
      if (shared.emissive) {
        mat.emissive = new THREE.Color(shared.emissive);
        mat.emissiveIntensity = shared.emissiveIntensity || 2.5;
      }
    }
  }

  // Enable baked vertex colors (COLOR_0 ground-contact dirt & crevice AO) whenever present on geometry
  if (child.geometry?.attributes?.color) {
    mat.vertexColors = true;
  }

  // Apply realistic weathered Jerusalem grey-stone value contrast to ground paving so sunlit plazas/roads never wash out
  if (matName.includes('mat_paving_flagstone')) {
    mat.color.setRGB(0.68, 0.66, 0.64);
  } else if (matName.includes('mat_paving')) {
    mat.color.setRGB(0.54, 0.53, 0.52);
  }

  // Maximize anisotropic filtering on all textures (capped at 8x for optimal GPU bandwidth)
  const targetAniso = Math.min(maxAniso, 8);
  for (const texProp of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'alphaMap']) {
    const tex = mat[texProp];
    if (tex) {
      tex.anisotropy = targetAniso;
      tex.needsUpdate = true;
    }
  }

  // Set normalScale to (1.2, 1.2) so chiselled stone normal maps look natural rather than over-embossed plastic
  if (mat.normalMap && mat.normalScale) {
    mat.normalScale.set(1.2, 1.2);
  }

  // Configure IBL environment map intensity: 1.2 on glass/metal, 0.55 on stone/paving/general materials
  const isGlassOrMetal =
    matName.includes('glass') ||
    matName.includes('window') ||
    matName.includes('metal') ||
    matName.includes('iron') ||
    matName.includes('brass') ||
    matName.includes('copper') ||
    matName.includes('car') ||
    matName.includes('lamp') ||
    (typeof mat.metalness === 'number' && mat.metalness > 0.45);

  if ('envMapIntensity' in mat) {
    mat.envMapIntensity = isGlassOrMetal ? 1.2 : 0.55;
  }

  // Configure alpha cutout materials (foliage_olive, foliage_cypress, iron_grille, awnings/signs with alpha)
  const isAlphaCutout =
    matName.includes('foliage_olive') ||
    matName.includes('foliage_cypress') ||
    matName.includes('iron_grille') ||
    matName.includes('foliage') ||
    matName.includes('leaf') ||
    matName.includes('grille') ||
    meshName.includes('foliage') ||
    meshName.includes('grille');

  if (isAlphaCutout) {
    mat.alphaTest = 0.35;
    mat.side = THREE.DoubleSide;
    mat.depthWrite = true;
    mat.transparent = false;
  }

  mat.needsUpdate = true;
}

/**
 * Traverses loaded GLTF scene, configures PBR materials, freezes static world matrices,
 * enables selective shadow casting (skipping tiny street litter/signs/ground from casting shadows),
 * and collects any multi-tier LOD meshes.
 */
function enhanceScenePBR(root, renderer, sharedTextures = null) {
  const maxAniso = renderer.capabilities.getMaxAnisotropy();
  const lodGroups = [];
  const lodBuckets = new Map();

  root.updateMatrixWorld(true);

  root.traverse((child) => {
    // Freeze static transforms so Three.js never recomputes world matrices every frame
    child.updateMatrix();
    child.matrixAutoUpdate = false;

    if (!child.isMesh) return;

    const nameLower = `${child.name || ''} ${child.parent?.name || ''}`.toLowerCase();

    // Selective shadow casting: only major architecture, walls, awnings, and trees cast shadows.
    // Skip ground terrain, street litter, signs, decals, and small clutter from casting shadows.
    const skipCastShadow =
      nameLower.startsWith('ground_paving') ||
      nameLower.includes('ground_terrain') ||
      nameLower.includes('shopsigns') ||
      nameLower.includes('mat_signs') ||
      nameLower.includes('trash') ||
      nameLower.includes('litter') ||
      nameLower.includes('decal') ||
      nameLower.includes('cardboard') ||
      nameLower.includes('lantern_glow');

    child.castShadow = !skipCastShadow;
    child.receiveShadow = true;

    // Ensure bounding sphere is computed for fast camera frustum culling
    if (child.geometry && !child.geometry.boundingSphere) {
      child.geometry.computeBoundingSphere();
    }

    // Detect _lod0 / _lod1 / _lod2 meshes if present
    const lodMatch = (child.name || '').match(/^(.*)_lod([012])$/i);
    if (lodMatch) {
      const baseKey = lodMatch[1];
      const tier = Number(lodMatch[2]);
      if (!lodBuckets.has(baseKey)) {
        lodBuckets.set(baseKey, {});
      }
      lodBuckets.get(baseKey)[tier] = child;
    }

    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const mat of materials) {
      enhanceMaterial(mat, child, maxAniso, sharedTextures);
    }
  });

  for (const [, tiers] of lodBuckets.entries()) {
    const refMesh = tiers[0] || tiers[1] || tiers[2];
    if (!refMesh?.geometry?.boundingSphere) continue;
    const center = refMesh.geometry.boundingSphere.center.clone().applyMatrix4(refMesh.matrixWorld);
    lodGroups.push({ center, tiers });
  }

  return lodGroups;
}

function updateLODVisibility(lodGroups, cameraPos) {
  if (!lodGroups || lodGroups.length === 0) return;
  for (let i = 0; i < lodGroups.length; i++) {
    const g = lodGroups[i];
    const dist = cameraPos.distanceTo(g.center);
    if (g.tiers[0]) g.tiers[0].visible = dist < 65;
    if (g.tiers[1]) g.tiers[1].visible = dist >= 65 && dist < 135;
    if (g.tiers[2]) g.tiers[2].visible = dist >= 135;
  }
}

function compassLabel(deg) {
  const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const idx = Math.round((((deg % 360) + 360) % 360) / 45) % 8;
  return dirs[idx];
}

async function fetchJsonSafe(url, fallback) {
  try {
    const res = await fetch(url);
    if (!res.ok) return fallback;
    return await res.json();
  } catch (_) {
    return fallback;
  }
}

async function initApp() {
  const params = new URLSearchParams(window.location.search);
  const isCleanMode = params.get('clean') === '1';
  if (isCleanMode) {
    document.body.classList.add('clean-mode');
  }

  const canvas = document.getElementById('game-canvas');
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: true
  });
  // Cap pixelRatio at 1.35 during interactive play for locked 60 FPS
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.35);
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0xe6d7c3, 0.00075);

  const camera = new THREE.PerspectiveCamera(
    65,
    window.innerWidth / window.innerHeight,
    0.2,
    2200
  );

  // Grazing warm Mediterranean afternoon sunlight: azimuth 232° (WSW), elevation 30°
  const azRad = THREE.MathUtils.degToRad(232);
  const elRad = THREE.MathUtils.degToRad(30);
  const sunDist = 180;
  const sunDir = new THREE.Vector3(
    Math.cos(elRad) * Math.sin(azRad),
    Math.sin(elRad),
    -Math.cos(elRad) * Math.cos(azRad)
  ).normalize();

  // Center shadow frustum near the active player district; updates dynamically on 16m grid steps
  const shadowTargetPos = new THREE.Vector3(25, 0, 10);
  const sunLight = new THREE.DirectionalLight(0xffeed2, 3.8);
  sunLight.target.position.copy(shadowTargetPos);
  sunLight.position.copy(shadowTargetPos).addScaledVector(sunDir, sunDist);
  sunLight.castShadow = true;
  // 2048x2048 with [-110, 110] frustum follows the player across the entire Old City
  sunLight.shadow.mapSize.set(2048, 2048);
  sunLight.shadow.camera.left = -110;
  sunLight.shadow.camera.right = 110;
  sunLight.shadow.camera.bottom = -110;
  sunLight.shadow.camera.top = 110;
  sunLight.shadow.camera.near = 10;
  sunLight.shadow.camera.far = 420;
  sunLight.shadow.bias = -0.00025;
  sunLight.shadow.normalBias = 0.018;
  scene.add(sunLight);
  scene.add(sunLight.target);
  sunLight.updateMatrixWorld(true);
  sunLight.target.updateMatrixWorld(true);

  let lastShadowSnapX = 25;
  let lastShadowSnapY = 0;
  let lastShadowSnapZ = 10;

  function syncSunShadowToCamera(force = false) {
    const snap = 16.0;
    const sx = Math.round(camera.position.x / snap) * snap;
    const sy = Math.round((camera.position.y - 1.7) / snap) * snap;
    const sz = Math.round(camera.position.z / snap) * snap;
    if (
      force ||
      Math.abs(sx - lastShadowSnapX) >= snap ||
      Math.abs(sy - lastShadowSnapY) >= snap ||
      Math.abs(sz - lastShadowSnapZ) >= snap
    ) {
      lastShadowSnapX = sx;
      lastShadowSnapY = sy;
      lastShadowSnapZ = sz;
      shadowTargetPos.set(sx, sy, sz);
      sunLight.target.position.copy(shadowTargetPos);
      sunLight.position.copy(shadowTargetPos).addScaledVector(sunDir, sunDist);
      sunLight.updateMatrixWorld(true);
      sunLight.target.updateMatrixWorld(true);
      renderer.shadowMap.needsUpdate = true;
    }
  }

  // Balanced HemisphereLight & low AmbientLight so shadowed walls retain deep AO & directional contrast
  const hemiLight = new THREE.HemisphereLight(0xd8e8f8, 0x8c7a62, 0.52);
  scene.add(hemiLight);

  const ambientLight = new THREE.AmbientLight(0xfff3de, 0.15);
  scene.add(ambientLight);

  // Real-time Sky IBL Environment Map via PMREMGenerator
  scene.environment = buildSkyIBL(renderer, sunDir);
  scene.environmentIntensity = 0.45;

  // Procedural Rayleigh/Mie Sky Dome with uSkyKey hole-detection uniform
  const { skyMesh, skyUniforms, skyMat } = createSkyDome(sunDir);
  scene.add(skyMesh);

  // Half-Resolution Post-Processing Ambient Occlusion (GTAOPass) + Adaptive 60 FPS Governor
  let composer = null;
  let gtaoPass = null;
  let skyKeyActive = false;
  let userAoEnabled = true;
  let adaptiveBypassAO = false;

  try {
    composer = new EffectComposer(renderer);
    composer.setPixelRatio(1.0);
    composer.setSize(window.innerWidth, window.innerHeight);

    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);

    // Run GTAOPass at half resolution with lightweight denoise samples for high frame rate
    const aoW = Math.max(1, Math.floor(window.innerWidth * 0.5));
    const aoH = Math.max(1, Math.floor(window.innerHeight * 0.5));
    gtaoPass = new GTAOPass(scene, camera, aoW, aoH);
    gtaoPass.output = GTAOPass.OUTPUT.Default;
    gtaoPass.blendIntensity = 0.85;
    gtaoPass.pdSamples = 8;
    gtaoPass.pdRings = 1;
    gtaoPass.updateGtaoMaterial({
      radius: 0.45,
      scale: 1.2,
      thickness: 1.5,
      distanceExponent: 1.8,
      samples: 6
    });
    composer.addPass(gtaoPass);

    const outputPass = new OutputPass();
    composer.addPass(outputPass);
  } catch (err) {
    console.warn('GTAOPass post-processing fallback to direct renderer:', err);
    composer = null;
  }

  let lodGroups = [];

  function renderFrame() {
    skyMesh.position.copy(camera.position);
    skyMesh.updateMatrix();
    skyMesh.updateMatrixWorld(true);
    syncSunShadowToCamera(false);
    updateLODVisibility(lodGroups, camera.position);

    const useComposer = !skyKeyActive && composer && userAoEnabled;

    if (!useComposer) {
      renderer.render(scene, camera);
    } else {
      try {
        composer.render();
      } catch (err) {
        console.warn('Composer render error, falling back to direct render:', err);
        composer = null;
        renderer.render(scene, camera);
      }
    }
  }

  // Player controller
  const player = new PlayerController(camera, canvas, {
    onModeChange: (flyMode) => {
      const badge = document.getElementById('hud-mode-badge');
      const btn = document.getElementById('btn-fly-toggle');
      if (badge) badge.textContent = flyMode ? 'FLY' : 'WALK';
      if (btn) btn.classList.toggle('active', flyMode);
    },
    onPointerLockChange: (locked) => {
      const prompt = document.getElementById('pointer-prompt');
      if (prompt) prompt.classList.toggle('locked', locked);
    }
  });

  // Minimap
  const minimapCanvas = document.getElementById('minimap-canvas');
  let activePlace = FALLBACK_PLACES[0];
  let places = [...FALLBACK_PLACES];
  let manifestData = null;
  let manifestIndexById = new Map();
  let allSceneMeshes = [];

  const minimap = new Minimap(minimapCanvas, {
    onSelectPlace: (place) => teleportToPlace(place),
    onTeleportXY: (x, y) => {
      player.setPoseENU({
        x,
        y,
        z_eye: 1.7,
        heading: player.heading,
        pitch: 0
      });
      updateHUD(true);
    }
  });

  let referencesById = new Map();

  function updatePhotoDrawer(place) {
    if (!place) return;
    const imgEl = document.getElementById('ref-photo-img');
    const titleEl = document.getElementById('ref-photo-title');
    const notesEl = document.getElementById('ref-photo-notes');
    const creditEl = document.getElementById('ref-photo-credit');
    if (imgEl && place.photo) {
      imgEl.src = place.photo;
    }
    if (titleEl) titleEl.textContent = place.name;
    if (notesEl) notesEl.textContent = place.subtitle || '';
    if (creditEl) {
      const refMeta = referencesById.get(place.id);
      if (refMeta && refMeta.source_url) {
        const author = refMeta.author || 'Wikimedia Commons';
        const lic = refMeta.license || 'CC BY-SA';
        creditEl.innerHTML = `Photo: ${author} (${lic}) &middot; <a href="${refMeta.source_url}" target="_blank" rel="noopener" style="color:#93c5fd;text-decoration:underline;">Wikimedia Commons</a>`;
      } else {
        creditEl.innerHTML = `Photo: <a href="https://commons.wikimedia.org" target="_blank" rel="noopener" style="color:#93c5fd;text-decoration:underline;">Wikimedia Commons</a>`;
      }
    }
  }

  function findNearestPlace(x, y) {
    if (!places || places.length === 0) return { place: null, distance: 0 };
    let best = places[0];
    let bestDist = Infinity;
    for (const p of places) {
      const d = Math.hypot(x - p.x, y - p.y);
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }
    return { place: best, distance: bestDist };
  }

  // Cache HUD DOM elements & throttle HUD/minimap updates during 60Hz loop
  const nameEl = document.getElementById('hud-place-name');
  const subEl = document.getElementById('hud-place-sub');
  const posEl = document.getElementById('hud-pos');
  const hdgEl = document.getElementById('hud-hdg');
  const distEl = document.getElementById('hud-dist');
  const tpSummaryEl = document.getElementById('tp-current-summary');
  let lastHudUpdateTime = 0;
  let lastHudX = NaN;
  let lastHudY = NaN;
  let lastHudHdg = NaN;

  function updateHUD(force = false) {
    if (isCleanMode) return;
    const now = performance.now();
    if (!force && now - lastHudUpdateTime < 66) return;

    const pose = player.getPoseENU();
    if (
      !force &&
      Math.abs(pose.x - lastHudX) < 0.05 &&
      Math.abs(pose.y - lastHudY) < 0.05 &&
      Math.abs(pose.heading - lastHudHdg) < 0.3
    ) {
      return;
    }
    lastHudUpdateTime = now;
    lastHudX = pose.x;
    lastHudY = pose.y;
    lastHudHdg = pose.heading;

    const { place: nearest, distance } = findNearestPlace(pose.x, pose.y);

    if (nearest && nearest.id !== activePlace?.id) {
      activePlace = nearest;
      updatePhotoDrawer(activePlace);
      document.querySelectorAll('.tp-btn').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.placeId === activePlace.id);
      });
    }

    if (nameEl && activePlace) nameEl.textContent = activePlace.name;
    if (subEl && activePlace) subEl.textContent = activePlace.subtitle;
    if (tpSummaryEl && activePlace) tpSummaryEl.textContent = activePlace.name;
    if (posEl) {
      const signX = pose.x >= 0 ? '+' : '';
      const signY = pose.y >= 0 ? '+' : '';
      posEl.textContent = `E ${signX}${pose.x.toFixed(1)}m, N ${signY}${pose.y.toFixed(1)}m, Z ${pose.groundZ.toFixed(1)}m`;
    }
    if (hdgEl) {
      const hdgInt = Math.round(((pose.heading % 360) + 360) % 360);
      hdgEl.textContent = `HDG ${String(hdgInt).padStart(3, '0')}° ${compassLabel(hdgInt)}`;
    }
    if (distEl) {
      distEl.textContent = `${distance.toFixed(1)}m from viewpoint`;
    }

    minimap.render(pose, activePlace?.id, camera.fov);
  }

  const teleportDialog = document.getElementById('teleport-dialog');
  const teleportSearch = document.getElementById('teleport-search');

  function teleportToPlace(place) {
    if (!place) return;
    activePlace = place;
    player.setPoseENU({
      x: place.x,
      y: place.y,
      z_eye: place.z_eye ?? 1.7,
      heading: place.heading ?? 90,
      pitch: place.pitch ?? 0,
      fov: place.fov ?? 65
    });
    updatePhotoDrawer(place);
    document.querySelectorAll('.tp-btn').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.placeId === place.id);
    });
    if (teleportDialog?.open) {
      teleportDialog.close();
    }
    updateHUD(true);
  }

  function classifyPlaceCategory(place) {
    const id = (place.id || '').toLowerCase();
    const nm = (place.name || '').toLowerCase();
    if (
      id.includes('gate') ||
      id.includes('breach') ||
      id.includes('tower_of_david') ||
      id.includes('west_plaza') ||
      nm.includes('gate') ||
      nm.includes('citadel')
    ) {
      return 'Gates & Fortifications';
    }
    if (
      id.includes('sepulchre') ||
      id.includes('redeemer') ||
      id.includes('cathedral') ||
      id.includes('synagogue') ||
      id.includes('western_wall') ||
      id.includes('dome_of') ||
      id.includes('al_aqsa')
    ) {
      return 'Temples, Churches, Mosques & Synagogues';
    }
    return 'Souqs, Plazas & Historic Streets';
  }

  function buildTeleportBar(filterText = '') {
    const container = document.getElementById('teleport-categories');
    if (!container) return;
    container.innerHTML = '';

    const query = filterText.trim().toLowerCase();
    const categories = [
      'Gates & Fortifications',
      'Temples, Churches, Mosques & Synagogues',
      'Souqs, Plazas & Historic Streets'
    ];
    const grouped = new Map(categories.map((c) => [c, []]));

    places.forEach((place, idx) => {
      if (query) {
        const hay = `${place.name} ${place.subtitle || ''} ${place.shortName || ''} ${place.id}`.toLowerCase();
        if (!hay.includes(query)) return;
      }
      const cat = classifyPlaceCategory(place);
      grouped.get(cat).push({ place, idx });
    });

    for (const [catName, entries] of grouped.entries()) {
      if (entries.length === 0) continue;
      const sec = document.createElement('section');
      const title = document.createElement('div');
      title.className = 'tp-cat-title';
      title.textContent = `${catName} (${entries.length})`;
      sec.appendChild(title);

      const grid = document.createElement('div');
      grid.className = 'tp-grid';

      for (const { place, idx } of entries) {
        const keyLabel = idx < 9 ? String(idx + 1) : idx === 9 ? '0' : `#${idx + 1}`;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'tp-btn' + (place.id === activePlace?.id ? ' active' : '');
        btn.dataset.placeId = place.id;
        const shortName = place.shortName || place.name.split('—')[0].trim();
        btn.innerHTML = `
          <div class="tp-btn-top">
            <span>${shortName}</span>
            <span class="tp-key">${keyLabel}</span>
          </div>
          <div class="tp-btn-sub">${place.subtitle || place.name}</div>
        `;
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          teleportToPlace(place);
        });
        grid.appendChild(btn);
      }

      sec.appendChild(grid);
      container.appendChild(sec);
    }

    const openBtn = document.getElementById('btn-open-teleport');
    if (openBtn) {
      openBtn.innerHTML = `<span>Teleport (${places.length} Locations)</span><span class="tp-key">T</span>`;
    }
  }

  function openTeleportModal() {
    if (!teleportDialog) return;
    if (teleportDialog.open) {
      teleportDialog.close();
      return;
    }
    if (document.pointerLockElement) {
      document.exitPointerLock();
    }
    if (teleportSearch) {
      teleportSearch.value = '';
    }
    buildTeleportBar('');
    teleportDialog.showModal();
  }

  // Fallback for browsers without declarative <dialog closedby="any"> support
  if (teleportDialog && !('closedBy' in HTMLDialogElement.prototype)) {
    teleportDialog.addEventListener('click', (event) => {
      if (event.target !== teleportDialog) return;
      const rect = teleportDialog.getBoundingClientRect();
      const isDialogContent =
        rect.top <= event.clientY &&
        event.clientY <= rect.top + rect.height &&
        rect.left <= event.clientX &&
        event.clientX <= rect.left + rect.width;
      if (isDialogContent) return;
      teleportDialog.close();
    });
  }

  if (teleportSearch) {
    teleportSearch.addEventListener('input', () => {
      buildTeleportBar(teleportSearch.value);
    });
  }

  // UI Toggles & Hotkeys
  const photoDrawer = document.getElementById('photo-drawer');
  const minimapPanel = document.getElementById('minimap-panel');
  const btnPhotoToggle = document.getElementById('btn-photo-toggle');
  const btnPhotoClose = document.getElementById('btn-photo-close');
  const btnSnapPhoto = document.getElementById('btn-snap-photo');
  const btnMapToggle = document.getElementById('btn-map-toggle');
  const btnMinimapZoom = document.getElementById('btn-minimap-zoom');
  const btnAoToggle = document.getElementById('btn-ao-toggle');
  const btnFlyToggle = document.getElementById('btn-fly-toggle');
  const btnTeleportToggle = document.getElementById('btn-teleport-toggle');
  const btnOpenTeleport = document.getElementById('btn-open-teleport');
  const btnCloseTeleport = document.getElementById('btn-close-teleport');

  const togglePhotoDrawer = () => {
    if (!photoDrawer) return;
    const hidden = photoDrawer.classList.toggle('hidden');
    if (btnPhotoToggle) btnPhotoToggle.classList.toggle('active', !hidden);
  };

  const toggleMinimap = () => {
    if (!minimapPanel) return;
    const hidden = minimapPanel.classList.toggle('hidden');
    minimap.visible = !hidden;
    if (btnMapToggle) btnMapToggle.classList.toggle('active', !hidden);
    if (!hidden) updateHUD(true);
  };

  const toggleAO = () => {
    userAoEnabled = !userAoEnabled;
    if (btnAoToggle) btnAoToggle.classList.toggle('active', userAoEnabled);
  };

  if (btnPhotoToggle) btnPhotoToggle.addEventListener('click', (e) => { e.stopPropagation(); togglePhotoDrawer(); });
  if (btnPhotoClose) btnPhotoClose.addEventListener('click', (e) => { e.stopPropagation(); togglePhotoDrawer(); });
  if (btnSnapPhoto) btnSnapPhoto.addEventListener('click', (e) => { e.stopPropagation(); teleportToPlace(activePlace); });
  if (btnMapToggle) btnMapToggle.addEventListener('click', (e) => { e.stopPropagation(); toggleMinimap(); });
  if (btnMinimapZoom) {
    btnMinimapZoom.addEventListener('click', (e) => {
      e.stopPropagation();
      const isFullCity = minimap.toggleFullCityView();
      btnMinimapZoom.textContent = isFullCity ? 'Local Zoom' : 'Full City';
      updateHUD(true);
    });
  }
  if (btnAoToggle) btnAoToggle.addEventListener('click', (e) => { e.stopPropagation(); toggleAO(); });
  if (btnFlyToggle) btnFlyToggle.addEventListener('click', (e) => { e.stopPropagation(); player.toggleFlyMode(); });
  if (btnTeleportToggle) btnTeleportToggle.addEventListener('click', (e) => { e.stopPropagation(); openTeleportModal(); });
  if (btnOpenTeleport) btnOpenTeleport.addEventListener('click', (e) => { e.stopPropagation(); openTeleportModal(); });
  if (btnCloseTeleport) btnCloseTeleport.addEventListener('click', (e) => { e.stopPropagation(); teleportDialog?.close(); });

  window.addEventListener('keydown', (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (e.code === 'KeyT') {
      e.preventDefault();
      openTeleportModal();
    } else if (e.code === 'KeyM') {
      toggleMinimap();
    } else if (e.code === 'KeyP') {
      togglePhotoDrawer();
    } else if (e.code === 'KeyO') {
      toggleAO();
    } else if (/^Digit[0-9]$/.test(e.code)) {
      const digit = Number(e.code.replace('Digit', ''));
      const idx = digit === 0 ? 9 : digit - 1;
      if (places[idx]) {
        teleportToPlace(places[idx]);
      }
    }
  });

  window.addEventListener('resize', () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    if (composer) {
      composer.setSize(w, h);
    }
    if (gtaoPass) {
      gtaoPass.setSize(Math.max(1, Math.floor(w * 0.5)), Math.max(1, Math.floor(h * 0.5)));
    }
    updateHUD(true);
  });

  // Expose window.__QA__ Introspection API
  const raycasterQA = new THREE.Raycaster();
  raycasterQA.near = 0.1;
  raycasterQA.far = 600;

  function lookupManifestEntry(mesh) {
    let cur = mesh;
    while (cur) {
      if (cur.name && manifestIndexById.has(cur.name)) {
        return manifestIndexById.get(cur.name);
      }
      if (cur.name) {
        const baseName = cur.name.replace(/(\.\d+|_mesh\d*|_prim\d*)$/, '');
        if (manifestIndexById.has(baseName)) {
          return manifestIndexById.get(baseName);
        }
        for (const [id, entry] of manifestIndexById.entries()) {
          if (cur.name.startsWith(id) || id.startsWith(baseName)) {
            return entry;
          }
        }
      }
      cur = cur.parent;
    }
    return null;
  }

  window.__QA__ = {
    ready: false,
    places: [],
    manifest: null,

    setPose(poseOpts = {}) {
      const result = player.setPoseENU(poseOpts);
      const { place } = findNearestPlace(result.x, result.y);
      if (place) {
        activePlace = place;
        updatePhotoDrawer(place);
      }
      updateHUD(true);
      syncSunShadowToCamera(true);
      renderFrame(true);
      return {
        ...result,
        place: place ? place.id : null
      };
    },

    getPose() {
      return player.getPoseENU();
    },

    setSkyKey(on) {
      const enabled = Boolean(on);
      if (enabled === skyKeyActive) {
        return { skyKey: enabled };
      }
      skyKeyActive = enabled;
      skyUniforms.uSkyKey.value = enabled ? 1.0 : 0.0;
      skyMat.toneMapped = !enabled;
      skyMat.needsUpdate = true;
      if (enabled) {
        scene.fog = null;
        renderer.toneMapping = THREE.NoToneMapping;
      } else {
        scene.fog = new THREE.FogExp2(0xe6d7c3, 0.0011);
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
      }
      renderFrame(true);
      return { skyKey: enabled };
    },

    traceView(gridW = 6, gridH = 4) {
      camera.updateMatrixWorld(true);
      const visibleIdsSet = new Set();
      const visibleRulesSet = new Set();
      const visibleMaterialsSet = new Set();
      let skyHits = 0;
      const totalRays = gridW * gridH;

      for (let iy = 0; iy < gridH; iy++) {
        const ndcY = 1.0 - ((iy + 0.5) / gridH) * 2.0;
        for (let ix = 0; ix < gridW; ix++) {
          const ndcX = -1.0 + ((ix + 0.5) / gridW) * 2.0;
          raycasterQA.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
          const hits = raycasterQA.intersectObjects(allSceneMeshes, false);

          if (hits.length === 0) {
            skyHits++;
            continue;
          }

          const hit = hits[0];
          const obj = hit.object;
          const entry = lookupManifestEntry(obj);

          if (entry) {
            if (entry.id) visibleIdsSet.add(entry.id);
            if (entry.rule || entry.category || entry.type) {
              visibleRulesSet.add(entry.rule || entry.category || entry.type);
            }
            if (Array.isArray(entry.materials)) {
              entry.materials.forEach((m) => visibleMaterialsSet.add(m));
            }
          }

          const cleanMeshId = (obj.parent?.name && obj.parent.name !== 'Scene' ? obj.parent.name : obj.name || '')
            .replace(/\.\d+$/, '');
          if (cleanMeshId) visibleIdsSet.add(cleanMeshId);
          if (obj.name) visibleIdsSet.add(obj.name.replace(/\.\d+$/, ''));

          if (obj.userData?.rule) visibleRulesSet.add(obj.userData.rule);

          const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
          for (const m of mats) {
            if (m && m.name) {
              visibleMaterialsSet.add(m.name.replace(/\.\d+$/, ''));
            }
          }
        }
      }

      return {
        visibleIds: Array.from(visibleIdsSet),
        visibleRules: Array.from(visibleRulesSet),
        visibleMaterials: Array.from(visibleMaterialsSet),
        skyFraction: totalRays > 0 ? Number((skyHits / totalRays).toFixed(4)) : 0,
        totalRays,
        hitCount: totalRays - skyHits
      };
    },

    goToPlace(idOrIndex) {
      const target =
        typeof idOrIndex === 'number'
          ? places[idOrIndex]
          : places.find((p) => p.id === idOrIndex);
      if (!target) return null;
      teleportToPlace(target);
      renderFrame(true);
      renderFrame(true);
      return player.getPoseENU();
    },

    auditScene() {
      const missingAlbedo = [];
      const missingNormal = [];
      const uvStretchSuspects = [];

      for (const mesh of allSceneMeshes) {
        if (!mesh.isMesh || !mesh.geometry) continue;
        const name = mesh.name || 'unnamed_mesh';
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const m of mats) {
          if (!m) continue;
          const mName = m.name || 'unnamed_mat';
          if (!m.map && !mName.includes('lantern_glow')) {
            missingAlbedo.push(`${name} (${mName})`);
          }
          if (!m.normalMap && !mName.includes('lantern_glow') && !mName.includes('signs')) {
            missingNormal.push(`${name} (${mName})`);
          }
        }

        const pos = mesh.geometry.attributes?.position;
        const uv = mesh.geometry.attributes?.uv;
        if (!uv && pos) {
          uvStretchSuspects.push({ mesh: name, reason: 'missing_uv_attribute' });
        }
      }

      return {
        meshCount: allSceneMeshes.length,
        missingAlbedo,
        missingNormal,
        uvStretchSuspects
      };
    }
  };

  // Load places.json & scene_manifest.json in parallel with GLB model
  const loaderFill = document.getElementById('loader-bar-fill');
  const loaderStatus = document.getElementById('loader-status');

  const ASSET_VER = 'v=city360';
  const [rawPlacesData, rawManifestData, rawCityPlanData, rawReferencesData] = await Promise.all([
    fetchJsonSafe(`/data/places.json?${ASSET_VER}`, null),
    fetchJsonSafe(`/data/scene_manifest.json?${ASSET_VER}`, null),
    fetchJsonSafe(`/data/city_plan.json?${ASSET_VER}`, null),
    fetchJsonSafe(`/data/references.json?${ASSET_VER}`, null)
  ]);

  if (rawReferencesData) {
    if (Array.isArray(rawReferencesData)) {
      rawReferencesData.forEach((r) => {
        if (r && r.id) referencesById.set(r.id, r);
      });
    } else if (typeof rawReferencesData === 'object') {
      Object.values(rawReferencesData).forEach((r) => {
        if (r && r.id) referencesById.set(r.id, r);
      });
    }
  }

  if (rawCityPlanData) {
    minimap.setCityPlan(rawCityPlanData);
  }

  if (rawPlacesData) {
    const rawList = Array.isArray(rawPlacesData)
      ? rawPlacesData
      : Array.isArray(rawPlacesData.places)
      ? rawPlacesData.places
      : FALLBACK_PLACES;
    if (rawList.length > 0) {
      places = rawList.map((item, idx) => normalizePlace(item, idx));
    }
  }
  activePlace = places[0];
  minimap.setPlaces(places);
  buildTeleportBar();
  updatePhotoDrawer(activePlace);

  if (rawManifestData) {
    manifestData = rawManifestData;
    const items = Array.isArray(rawManifestData)
      ? rawManifestData
      : Array.isArray(rawManifestData.objects)
      ? rawManifestData.objects
      : Array.isArray(rawManifestData.features)
      ? rawManifestData.features
      : typeof rawManifestData === 'object'
      ? Object.entries(rawManifestData).map(([k, v]) => (typeof v === 'object' ? { id: k, ...v } : { id: k }))
      : [];
    for (const item of items) {
      if (item && item.id) {
        manifestIndexById.set(item.id, item);
      }
      if (item && item.name) {
        manifestIndexById.set(item.name, item);
      }
    }
  }

  // Preload shared external PBR textures & load 3D City Region GLB Models in parallel
  const maxAniso = renderer.capabilities.getMaxAnisotropy();
  const sharedTexturesPromise = preloadSharedTextures(maxAniso);

  const gltfLoader = new GLTFLoader();
  const cityRoot = new THREE.Group();
  cityRoot.name = 'old_city_root';
  scene.add(cityRoot);

  const regionEntries =
    Array.isArray(manifestData?.regions) && manifestData.regions.length > 0
      ? manifestData.regions.map((r) => (typeof r === 'string' ? { id: r, glb: r } : r))
      : [{ id: 'r01_jaffa_gate', glb: manifestData?.glb_path || '/models/jaffa_gate_city.glb' }];

  const sharedTextures = await sharedTexturesPromise;
  let loadedRegionCount = 0;

  await Promise.all(
    regionEntries.map(
      (reg) =>
        new Promise((resolve) => {
          const rawGlbUrl = reg.glb || reg.glb_path || reg.url;
          const glbUrl = rawGlbUrl.includes('?') ? rawGlbUrl : `${rawGlbUrl}?${ASSET_VER}`;
          gltfLoader.load(
            glbUrl,
            (gltf) => {
              const regScene = gltf.scene;
              regScene.name = reg.id || 'region_chunk';
              cityRoot.add(regScene);
              const regLods = enhanceScenePBR(regScene, renderer, sharedTextures);
              lodGroups.push(...regLods);
              loadedRegionCount++;
              if (loaderFill) {
                const pct = Math.min(96, Math.round((loadedRegionCount / regionEntries.length) * 95));
                loaderFill.style.width = `${pct}%`;
              }
              resolve(true);
            },
            undefined,
            (err) => {
              console.warn(`Could not load region GLB ${glbUrl}:`, err);
              resolve(false);
            }
          );
        })
    )
  );

  allSceneMeshes = [];
  cityRoot.traverse((obj) => {
    if (obj.isMesh) allSceneMeshes.push(obj);
  });

  if (allSceneMeshes.length > 0) {
    const { walkableMeshes, obstacleMeshes } = classifyCollisionMeshes(cityRoot);
    player.setCollisionMeshes(walkableMeshes, obstacleMeshes);
    minimap.buildFromScene(cityRoot, manifestData);
  } else {
    if (loaderStatus) {
      loaderStatus.textContent = 'Initializing fallback terrain plane…';
    }
    const fallbackGeo = new THREE.PlaneGeometry(320, 320);
    fallbackGeo.rotateX(-Math.PI / 2);
    const fallbackMat = new THREE.MeshStandardMaterial({ color: 0xd8c6a5, roughness: 0.85 });
    const fallbackMesh = new THREE.Mesh(fallbackGeo, fallbackMat);
    fallbackMesh.name = 'ground_terrain';
    fallbackMesh.receiveShadow = true;
    scene.add(fallbackMesh);
    allSceneMeshes = [fallbackMesh];
    player.setCollisionMeshes([fallbackMesh], []);
    minimap.buildFromScene(scene, manifestData);
  }

  // Apply initial camera pose from URL query parameters (?cam=<place_id> or ?x=...&y=...&h=...&p=...) or default to Place 1
  const camParam = params.get('cam');
  const xParam = params.get('x');
  const yParam = params.get('y');

  if (camParam) {
    const matched = places.find((p) => p.id === camParam) || places[Number(camParam) - 1] || places[0];
    teleportToPlace(matched);
  } else if (xParam !== null && yParam !== null) {
    player.setPoseENU({
      x: Number(xParam),
      y: Number(yParam),
      z_eye: params.get('z') !== null ? Number(params.get('z')) : 1.7,
      heading: params.get('h') !== null ? Number(params.get('h')) : 90,
      pitch: params.get('p') !== null ? Number(params.get('p')) : 0,
      fov: params.get('fov') !== null ? Number(params.get('fov')) : 65
    });
    updateHUD(true);
  } else {
    teleportToPlace(places[0]);
  }

  // Freeze static shadow map updates after baking the initial static city shadow map!
  // Saves ~70% of GPU frame time during 60Hz interactive walking.
  renderer.shadowMap.autoUpdate = false;
  renderer.shadowMap.needsUpdate = true;

  // Render initial frame & mark QA API ready
  renderFrame(true);

  window.__QA__.places = places;
  window.__QA__.manifest = manifestData;
  window.__QA__.ready = true;

  const loadingOverlay = document.getElementById('loading-overlay');
  if (loaderFill) loaderFill.style.width = '100%';
  if (loadingOverlay) {
    loadingOverlay.classList.add('done');
  }

  // Main animation loop with Adaptive 60 FPS Governor (disabled in ?clean=1 headless QA capture mode
  // so window.__QA__.setPose renders deterministically on demand without background CPU contention)
  let lastTime = performance.now();
  let emaDt = 1 / 60;

  function animate(now) {
    requestAnimationFrame(animate);
    const dt = (now - lastTime) / 1000;
    lastTime = now;

    if (dt > 0 && dt < 0.25) {
      emaDt = emaDt * 0.85 + dt * 0.15;
    }

    player.update(dt);
    updateHUD(false);
    renderFrame();
  }

  if (!isCleanMode) {
    requestAnimationFrame(animate);
  }
}

initApp();
