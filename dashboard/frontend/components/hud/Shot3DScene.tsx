"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { Instances, Instance, OrbitControls, type OrbitControlsProps } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

export type Shot = {
  x: number; // NHL coords: -100..100 (length, x=89 = goal line)
  y: number; // -42..42 (width)
  goal?: boolean;
  type?: string;
  value?: number;
};

export type Shot3DSceneProps = {
  shots: Shot[];
  themeColor?: string;
  flip?: boolean;
};

// NHL rink: 200×85 ft, corner radius 28 ft
const RINK_LEN = 200;
const RINK_WID = 85;
const CORNER_R = 28;

// ─── Rounded rink shape (used by both surface + boards) ────────────────────
function rinkShape(inset = 0): THREE.Shape {
  const halfLen = RINK_LEN / 2 - inset;
  const halfWid = RINK_WID / 2 - inset;
  const r = Math.max(0, CORNER_R - inset);
  const s = new THREE.Shape();
  s.moveTo(-halfLen + r, -halfWid);
  s.lineTo(halfLen - r, -halfWid);
  s.absarc(halfLen - r, -halfWid + r, r, -Math.PI / 2, 0, false);
  s.lineTo(halfLen, halfWid - r);
  s.absarc(halfLen - r, halfWid - r, r, 0, Math.PI / 2, false);
  s.lineTo(-halfLen + r, halfWid);
  s.absarc(-halfLen + r, halfWid - r, r, Math.PI / 2, Math.PI, false);
  s.lineTo(-halfLen, -halfWid + r);
  s.absarc(-halfLen + r, -halfWid + r, r, Math.PI, 1.5 * Math.PI, false);
  return s;
}

// ─── Boards — rounded extruded ring ────────────────────────────────────────
function Boards({ themeColor }: { themeColor: string }) {
  const c = useMemo(() => new THREE.Color(themeColor), [themeColor]);
  const outer = useMemo(() => rinkShape(0), []);
  const inner = useMemo(() => rinkShape(0.6), []);
  const shape = useMemo(() => {
    outer.holes = [new THREE.Path(inner.getPoints(64))];
    return outer;
  }, [outer, inner]);

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.5, 0]}>
      <extrudeGeometry args={[shape, { depth: 1.0, bevelEnabled: false, curveSegments: 24 }]} />
      <meshStandardMaterial color={c} emissive={c} emissiveIntensity={0.28} roughness={0.4} />
    </mesh>
  );
}

// ─── Ice surface — pale icy blue, faint reflection ─────────────────────────
function IceSurface() {
  const shape = useMemo(() => rinkShape(0.6), []);
  return (
    <group>
      {/* Solid icy ice base — black-tinted but with a cool tint and slight gloss */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <shapeGeometry args={[shape, 32]} />
        <meshPhysicalMaterial
          color="#0a1018"
          roughness={0.15}
          metalness={0.05}
          clearcoat={0.8}
          clearcoatRoughness={0.20}
          emissive="#1a3050"
          emissiveIntensity={0.08}
        />
      </mesh>

      {/* Cool-blue ice gloss layer — additive light reflection feel */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}>
        <shapeGeometry args={[shape, 32]} />
        <meshBasicMaterial color="#3b6ea5" transparent opacity={0.05} blending={THREE.AdditiveBlending} />
      </mesh>
    </group>
  );
}

// ─── Rink markings — center, blue, goal lines + ovals ──────────────────────
function RinkMarkings() {
  return (
    <group position={[0, 0.02, 0]}>
      {/* Center red line */}
      <RinkLine x1={0} z1={-RINK_WID / 2 + 2} x2={0} z2={RINK_WID / 2 - 2} color="#e63a3a" width={0.5} />
      {/* Blue lines */}
      <RinkLine x1={-25} z1={-RINK_WID / 2 + 2} x2={-25} z2={RINK_WID / 2 - 2} color="#3b8fd0" width={0.5} />
      <RinkLine x1={25}  z1={-RINK_WID / 2 + 2} x2={25}  z2={RINK_WID / 2 - 2} color="#3b8fd0" width={0.5} />
      {/* Goal lines (red) */}
      <RinkLine x1={-89} z1={-RINK_WID / 2 + 8} x2={-89} z2={RINK_WID / 2 - 8} color="#e63a3a" width={0.3} />
      <RinkLine x1={89}  z1={-RINK_WID / 2 + 8} x2={89}  z2={RINK_WID / 2 - 8} color="#e63a3a" width={0.3} />

      {/* Center red faceoff circle */}
      <FaceoffMarking x={0} z={0} radius={15} color="#3b8fd0" dotColor="#3b8fd0" />
      {/* End-zone faceoff circles (4 corners) — outside the blue line */}
      <FaceoffMarking x={-69} z={-22} radius={15} color="#e63a3a" dotColor="#e63a3a" hashes />
      <FaceoffMarking x={-69} z={22}  radius={15} color="#e63a3a" dotColor="#e63a3a" hashes />
      <FaceoffMarking x={69}  z={-22} radius={15} color="#e63a3a" dotColor="#e63a3a" hashes />
      <FaceoffMarking x={69}  z={22}  radius={15} color="#e63a3a" dotColor="#e63a3a" hashes />

      {/* Neutral zone faceoff dots (no circles, just dots) */}
      {[[-20, -22], [-20, 22], [20, -22], [20, 22]].map(([x, z], i) => (
        <FaceoffDot key={i} x={x} z={z} color="#e63a3a" />
      ))}

      {/* Creases */}
      <Crease x={-89} />
      <Crease x={89} />
    </group>
  );
}

function RinkLine({
  x1, z1, x2, z2, color, width,
}: { x1: number; z1: number; x2: number; z2: number; color: string; width: number }) {
  const cx = (x1 + x2) / 2;
  const cz = (z1 + z2) / 2;
  const dx = x2 - x1;
  const dz = z2 - z1;
  const len = Math.hypot(dx, dz);
  const angle = Math.atan2(dz, dx);
  return (
    <mesh position={[cx, 0, cz]} rotation={[-Math.PI / 2, 0, -angle]}>
      <planeGeometry args={[len, width]} />
      <meshBasicMaterial color={color} transparent opacity={0.75} />
    </mesh>
  );
}

function FaceoffMarking({
  x, z, radius, color, dotColor, hashes,
}: { x: number; z: number; radius: number; color: string; dotColor: string; hashes?: boolean }) {
  const ring = useMemo(() => {
    const pts: number[] = [];
    const segs = 60;
    const ringWidth = 0.30;
    // Tube-shaped ring (just two close circles)
    for (let i = 0; i <= segs; i++) {
      const a = (i / segs) * Math.PI * 2;
      pts.push(Math.cos(a) * radius, 0, Math.sin(a) * radius);
    }
    return { points: new Float32Array(pts), ringWidth };
  }, [radius]);

  return (
    <group position={[x, 0, z]}>
      {/* Painted ring — solid annular plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[radius - 0.25, radius + 0.25, 64]} />
        <meshBasicMaterial color={color} transparent opacity={0.70} />
      </mesh>
      {/* Center dot */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[1.0, 24]} />
        <meshBasicMaterial color={dotColor} transparent opacity={0.85} />
      </mesh>
      {/* Hash marks (interior + exterior) */}
      {hashes ? (
        <>
          {[[-2, 0], [2, 0]].map(([sx], i) => (
            <mesh key={`h-${i}`} position={[sx, 0, radius - 2]} rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry args={[0.4, 4]} />
              <meshBasicMaterial color={color} transparent opacity={0.55} />
            </mesh>
          ))}
          {[[-2, 0], [2, 0]].map(([sx], i) => (
            <mesh key={`h2-${i}`} position={[sx, 0, -(radius - 2)]} rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry args={[0.4, 4]} />
              <meshBasicMaterial color={color} transparent opacity={0.55} />
            </mesh>
          ))}
        </>
      ) : null}
    </group>
  );
}

function FaceoffDot({ x, z, color }: { x: number; z: number; color: string }) {
  return (
    <mesh position={[x, 0, z]} rotation={[-Math.PI / 2, 0, 0]}>
      <circleGeometry args={[0.8, 16]} />
      <meshBasicMaterial color={color} transparent opacity={0.80} />
    </mesh>
  );
}

function Crease({ x }: { x: number }) {
  const radius = 6;
  const sign = x < 0 ? 1 : -1;
  const points = useMemo(() => {
    const pts: number[] = [];
    const segs = 32;
    for (let i = 0; i <= segs; i++) {
      const a = -Math.PI / 2 + (i / segs) * Math.PI;
      pts.push(Math.cos(a) * radius * sign, 0, Math.sin(a) * radius);
    }
    return new Float32Array(pts);
  }, [sign]);

  return (
    <group position={[x, 0.005, 0]}>
      {/* Crease fill — light blue half-disc */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[radius, 24, sign > 0 ? Math.PI / 2 : -Math.PI / 2, Math.PI]} />
        <meshBasicMaterial color="#3b8fd0" transparent opacity={0.15} />
      </mesh>
      {/* Crease outline */}
      <line>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[points, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#e63a3a" transparent opacity={0.70} />
      </line>
    </group>
  );
}

// ─── 3D net — frame + mesh ─────────────────────────────────────────────────
function Net({ x }: { x: number }) {
  const sign = x < 0 ? -1 : 1;
  const netDepth = 3.5;
  const netWidth = 6;
  const netHeight = 4;
  const offset = sign * netDepth / 2;

  return (
    <group position={[x + offset, 0, 0]}>
      {/* Crossbar (top, horizontal across goal mouth) */}
      <mesh position={[-offset, netHeight, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.18, 0.18, netWidth, 8]} />
        <meshStandardMaterial color="#ff3030" emissive="#ff3030" emissiveIntensity={0.45} />
      </mesh>
      {/* Two side posts (vertical) */}
      <mesh position={[-offset, netHeight / 2, -netWidth / 2]}>
        <cylinderGeometry args={[0.18, 0.18, netHeight, 8]} />
        <meshStandardMaterial color="#ff3030" emissive="#ff3030" emissiveIntensity={0.45} />
      </mesh>
      <mesh position={[-offset, netHeight / 2, netWidth / 2]}>
        <cylinderGeometry args={[0.18, 0.18, netHeight, 8]} />
        <meshStandardMaterial color="#ff3030" emissive="#ff3030" emissiveIntensity={0.45} />
      </mesh>

      {/* Net mesh — semi-transparent box-ish frame behind the goal mouth */}
      <mesh position={[0, netHeight / 2, 0]}>
        <boxGeometry args={[netDepth, netHeight, netWidth]} />
        <meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.32} />
      </mesh>
      {/* Net inner shading */}
      <mesh position={[0, netHeight / 2, 0]}>
        <boxGeometry args={[netDepth - 0.1, netHeight - 0.1, netWidth - 0.1]} />
        <meshBasicMaterial color="#cce0ff" transparent opacity={0.06} />
      </mesh>
    </group>
  );
}

// ─── Shot instancing ───────────────────────────────────────────────────────
function ShotInstances({ shots, themeColor, flip }: { shots: Shot[]; themeColor: string; flip: boolean }) {
  const misses = useMemo(() => shots.filter((s) => !s.goal), [shots]);
  const goals = useMemo(() => shots.filter((s) => s.goal), [shots]);
  const fx = (s: Shot) => (flip ? -s.x : s.x);

  return (
    <group>
      {misses.length > 0 ? (
        <Instances limit={Math.max(1, misses.length)} range={misses.length}>
          <sphereGeometry args={[1.0, 12, 12]} />
          <meshStandardMaterial color={themeColor} emissive={themeColor} emissiveIntensity={0.40} roughness={0.4} />
          {misses.map((s, i) => (
            <Instance key={i} position={[fx(s), 0.8, s.y]} scale={0.9} />
          ))}
        </Instances>
      ) : null}
      {goals.length > 0 ? (
        <Instances limit={Math.max(1, goals.length)} range={goals.length}>
          <sphereGeometry args={[1.3, 14, 14]} />
          <meshStandardMaterial
            color="#ffffff"
            emissive={themeColor}
            emissiveIntensity={1.4}
            metalness={0.4}
            roughness={0.2}
          />
          {goals.map((s, i) => (
            <Instance key={i} position={[fx(s), 1.0, s.y]} scale={1.2} />
          ))}
        </Instances>
      ) : null}
    </group>
  );
}

// ─── Auto-rotate intro (2× CCW then stop) ──────────────────────────────────
function AutoRotateIntro({ controlsRef, active }: { controlsRef: React.RefObject<{ autoRotate: boolean; autoRotateSpeed: number } | null>; active: boolean }) {
  const start = useRef<number | null>(null);
  useFrame(() => {
    if (!active || !controlsRef.current) return;
    if (start.current === null) start.current = performance.now();
    const elapsed = (performance.now() - start.current) / 1000;
    // 2 full CCW rotations over 16s, then stop
    if (elapsed >= 16) {
      controlsRef.current.autoRotate = false;
    }
  });
  return null;
}

export default function Shot3DScene({ shots, themeColor = "#C9A84C", flip = false }: Shot3DSceneProps) {
  const [freePan, setFreePan] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [legendOpen, setLegendOpen] = useState(false);
  const controlsRef = useRef<OrbitControlsProps & { autoRotate?: boolean; autoRotateSpeed?: number } | null>(null);
  const dpr = typeof window !== "undefined" ? Math.min(window.devicePixelRatio, 1.5) : 1;

  // Stop auto-rotate after 16s
  useEffect(() => {
    if (!autoRotate) return;
    const t = setTimeout(() => setAutoRotate(false), 16000);
    return () => clearTimeout(t);
  }, [autoRotate]);

  return (
    <div className="relative w-full" style={{ aspectRatio: "200 / 120" }}>
      <Canvas
        frameloop={autoRotate ? "always" : "demand"}
        dpr={dpr}
        camera={{ position: [0, 70, 105], fov: 38, near: 1, far: 800 }}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        style={{ background: "transparent", touchAction: "none", cursor: freePan ? "move" : "grab" }}
      >
        {/* Cool icy lighting — blue ambient + cool directional */}
        <ambientLight intensity={0.55} color="#9bbcff" />
        <directionalLight position={[40, 80, 30]} intensity={0.75} color="#e0eaff" />
        <directionalLight position={[-30, 50, -20]} intensity={0.35} color="#80a8d8" />
        <hemisphereLight args={[0xbcd8ff, 0x0a1018, 0.35]} />

        <Boards themeColor={themeColor} />
        <IceSurface />
        <RinkMarkings />
        <Net x={-89} />
        <Net x={89} />
        <ShotInstances shots={shots} themeColor={themeColor} flip={flip} />

        <OrbitControls
          // @ts-expect-error drei typing — accept assignable to React ref
          ref={controlsRef}
          enableDamping
          dampingFactor={0.08}
          enablePan={freePan}
          enableZoom
          zoomSpeed={0.6}
          minDistance={40}
          maxDistance={250}
          minPolarAngle={0.05}
          maxPolarAngle={Math.PI / 2.02}
          autoRotate={autoRotate}
          autoRotateSpeed={-2.2}
        />
      </Canvas>

      <div className="absolute top-2 right-2 flex flex-col items-end gap-1">
        <button
          type="button"
          onClick={() => { setFreePan((v) => !v); setAutoRotate(false); }}
          className="hud-mono text-[9px] uppercase tracking-[0.18em] px-2 py-1 rounded border"
          style={{
            color: freePan ? "#4ade80" : themeColor,
            borderColor: freePan ? "rgba(74,222,128,0.55)" : `${themeColor}55`,
            background: "rgba(0,0,0,0.55)",
            backdropFilter: "blur(6px)",
          }}
        >
          {freePan ? "● unlocked · click to lock" : "◆ locked · click to free-pan"}
        </button>
        <span className="hud-mono text-[8px] uppercase tracking-[0.18em] px-1.5 py-0.5 rounded text-[var(--text-secondary)]"
          style={{ background: "rgba(0,0,0,0.45)" }}>
          drag · scroll · right-drag to pan
        </span>
      </div>

      {/* Glossary panel — collapsible, mobile-friendly */}
      <div className="absolute bottom-2 left-2 max-w-[80%]">
        <button
          type="button"
          onClick={() => setLegendOpen(v => !v)}
          aria-expanded={legendOpen}
          className="hud-mono text-[9px] uppercase tracking-[0.18em] px-2 py-1 rounded border flex items-center gap-1.5"
          style={{
            color: themeColor,
            borderColor: `${themeColor}55`,
            background: "rgba(0,0,0,0.65)",
            backdropFilter: "blur(6px)",
          }}
        >
          ◢ LEGEND
          <span className="opacity-60" aria-hidden>{legendOpen ? "▾" : "▸"}</span>
        </button>
        {legendOpen && (
          <div className="mt-1 hud-mono text-[8px] uppercase tracking-[0.16em] rounded border px-2 py-1.5"
            style={{
              color: "var(--text-secondary)",
              borderColor: `${themeColor}33`,
              background: "rgba(0,0,0,0.78)",
              backdropFilter: "blur(6px)",
            }}>
            <div className="flex items-center gap-1.5">
              <span className="inline-block rounded-full shrink-0" style={{ width: 8, height: 8, background: themeColor, boxShadow: `0 0 6px ${themeColor}` }} />
              <span>shot · miss</span>
            </div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="inline-block rounded-full shrink-0"
                style={{ width: 10, height: 10, background: "#fff", boxShadow: `0 0 10px ${themeColor}, inset 0 0 4px ${themeColor}` }} />
              <span>goal · scored</span>
            </div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="inline-block shrink-0" style={{ width: 8, height: 3, background: "#ff3030", borderRadius: 1, boxShadow: "0 0 4px rgba(255,48,48,0.6)" }} />
              <span>net · 6ft × 4ft</span>
            </div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="inline-block shrink-0" style={{ width: 8, height: 1.5, background: "#3b8fd0" }} />
              <span>blue line · OZ entry</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
