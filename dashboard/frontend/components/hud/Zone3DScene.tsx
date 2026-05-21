"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Text, Billboard } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

export type ZoneActivation = {
  slot: number;
  perim: number;
  net: number;
  cornerL: number;
  cornerR: number;
};

export type Zone3DSceneProps = {
  activations: ZoneActivation;
  themeColor?: string;
};

// Full NHL rink: 200×85 ft, corner radius 28 ft
const RINK_LEN = 200;
const RINK_WID = 85;
const CORNER_R = 28;

// ─── Rounded rink shape (matches Shot3DScene exactly) ──────────────────────
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

function IceSurface() {
  const shape = useMemo(() => rinkShape(0.6), []);
  return (
    <group>
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
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}>
        <shapeGeometry args={[shape, 32]} />
        <meshBasicMaterial color="#3b6ea5" transparent opacity={0.05} blending={THREE.AdditiveBlending} />
      </mesh>
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
  return (
    <group position={[x, 0, z]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[radius - 0.25, radius + 0.25, 64]} />
        <meshBasicMaterial color={color} transparent opacity={0.70} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[1.0, 24]} />
        <meshBasicMaterial color={dotColor} transparent opacity={0.85} />
      </mesh>
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
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[radius, 24, sign > 0 ? Math.PI / 2 : -Math.PI / 2, Math.PI]} />
        <meshBasicMaterial color="#3b8fd0" transparent opacity={0.15} />
      </mesh>
      <line>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[points, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#e63a3a" transparent opacity={0.70} />
      </line>
    </group>
  );
}

function RinkMarkings() {
  return (
    <group position={[0, 0.02, 0]}>
      <RinkLine x1={0} z1={-RINK_WID / 2 + 2} x2={0} z2={RINK_WID / 2 - 2} color="#e63a3a" width={0.5} />
      <RinkLine x1={-25} z1={-RINK_WID / 2 + 2} x2={-25} z2={RINK_WID / 2 - 2} color="#3b8fd0" width={0.5} />
      <RinkLine x1={25}  z1={-RINK_WID / 2 + 2} x2={25}  z2={RINK_WID / 2 - 2} color="#3b8fd0" width={0.5} />
      <RinkLine x1={-89} z1={-RINK_WID / 2 + 8} x2={-89} z2={RINK_WID / 2 - 8} color="#e63a3a" width={0.3} />
      <RinkLine x1={89}  z1={-RINK_WID / 2 + 8} x2={89}  z2={RINK_WID / 2 - 8} color="#e63a3a" width={0.3} />
      <FaceoffMarking x={0} z={0} radius={15} color="#3b8fd0" dotColor="#3b8fd0" />
      <FaceoffMarking x={-69} z={-22} radius={15} color="#e63a3a" dotColor="#e63a3a" hashes />
      <FaceoffMarking x={-69} z={22}  radius={15} color="#e63a3a" dotColor="#e63a3a" hashes />
      <FaceoffMarking x={69}  z={-22} radius={15} color="#e63a3a" dotColor="#e63a3a" hashes />
      <FaceoffMarking x={69}  z={22}  radius={15} color="#e63a3a" dotColor="#e63a3a" hashes />
      {[[-20, -22], [-20, 22], [20, -22], [20, 22]].map(([x, z], i) => (
        <FaceoffDot key={i} x={x} z={z} color="#e63a3a" />
      ))}
      <Crease x={-89} />
      <Crease x={89} />
    </group>
  );
}

function Net({ x }: { x: number }) {
  const sign = x < 0 ? -1 : 1;
  const netDepth = 3.5;
  const netWidth = 6;
  const netHeight = 4;
  const offset = sign * netDepth / 2;
  return (
    <group position={[x + offset, 0, 0]}>
      <mesh position={[-offset, netHeight, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.18, 0.18, netWidth, 8]} />
        <meshStandardMaterial color="#ff3030" emissive="#ff3030" emissiveIntensity={0.45} />
      </mesh>
      <mesh position={[-offset, netHeight / 2, -netWidth / 2]}>
        <cylinderGeometry args={[0.18, 0.18, netHeight, 8]} />
        <meshStandardMaterial color="#ff3030" emissive="#ff3030" emissiveIntensity={0.45} />
      </mesh>
      <mesh position={[-offset, netHeight / 2, netWidth / 2]}>
        <cylinderGeometry args={[0.18, 0.18, netHeight, 8]} />
        <meshStandardMaterial color="#ff3030" emissive="#ff3030" emissiveIntensity={0.45} />
      </mesh>
      <mesh position={[0, netHeight / 2, 0]}>
        <boxGeometry args={[netDepth, netHeight, netWidth]} />
        <meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.32} />
      </mesh>
      <mesh position={[0, netHeight / 2, 0]}>
        <boxGeometry args={[netDepth - 0.1, netHeight - 0.1, netWidth - 0.1]} />
        <meshBasicMaterial color="#cce0ff" transparent opacity={0.06} />
      </mesh>
    </group>
  );
}

// ─── Flat zone decals — colored discs on the ice with floating % labels ────
type Zone = {
  id: keyof ZoneActivation;
  cx: number;
  cz: number;
  w: number;
  d: number;
  baseColor: string;
  label: string;
};

function ZonePatch({
  zone, pct, hovered, onHover, onLeave,
}: {
  zone: Zone;
  pct: number;
  hovered: boolean;
  onHover: () => void;
  onLeave: () => void;
}) {
  // Color saturation tracks activity %
  const color = pct > 25 ? "#f87171" : pct > 15 ? "#fbbf24" : zone.baseColor;
  const alpha = Math.min(0.65, 0.15 + (pct / 35) * 0.55);

  // Per-zone label offset (y = height above ice). Spreading these so the five
  // chips don't collide in screen space when the camera looks from oblique
  // angles. L CORNER + R CORNER sit higher because they're outermost; SLOT
  // and NET sit at differentiated heights to avoid stacking near the crease.
  const labelY: Record<string, number> = {
    cornerL: 10,
    cornerR: 10,
    slot:    7,
    net:     4,
    perim:   8,
  };
  const y = labelY[zone.id] ?? 6;
  // Shrink the most word-heavy labels so they don't sprawl across neighbours.
  const isLong = zone.label.length > 4;

  return (
    <group position={[zone.cx, 0.04, zone.cz]}>
      {/* Flat colored decal */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        onPointerEnter={(e) => { e.stopPropagation(); onHover(); }}
        onPointerLeave={onLeave}
      >
        <planeGeometry args={[zone.w, zone.d]} />
        <meshBasicMaterial color={color} transparent opacity={hovered ? alpha * 1.4 : alpha} />
      </mesh>
      {/* Billboarded label group — always faces camera so the % and zone name
          stay legible no matter how the rink is rotated, and they keep a
          consistent screen-space size between neighbouring chips. */}
      <Billboard position={[0, y, 0]} follow lockX={false} lockY={false} lockZ={false}>
        <Text
          position={[0, 1.4, 0]}
          fontSize={2.8}
          color={color}
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.10}
          outlineColor="#000"
        >
          {`${pct.toFixed(0)}%`}
        </Text>
        <Text
          position={[0, -0.6, 0]}
          fontSize={isLong ? 1.0 : 1.3}
          color={color}
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.05}
          outlineColor="#000"
          letterSpacing={0.12}
        >
          {zone.label}
        </Text>
      </Billboard>
    </group>
  );
}

export default function Zone3DScene({ activations, themeColor = "#C9A84C" }: Zone3DSceneProps) {
  const [hover, setHover] = useState<keyof ZoneActivation | null>(null);
  const [freePan, setFreePan] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [legendOpen, setLegendOpen] = useState(false);
  const controlsRef = useRef<unknown>(null);
  const dpr = typeof window !== "undefined" ? Math.min(window.devicePixelRatio, 1.5) : 1;

  useEffect(() => {
    if (!autoRotate) return;
    const t = setTimeout(() => setAutoRotate(false), 16000);
    return () => clearTimeout(t);
  }, [autoRotate]);

  // Zones live in the OZ (right side of full rink, x > 25 = past blue line, x = 89 goal line)
  // Layout (full rink coords):
  //   - perim: behind blue line, wide strip
  //   - cornerL: top-side corner (z < 0)
  //   - cornerR: bottom-side corner (z > 0)
  //   - slot: central, between dots
  //   - net: tight band right in front of crease
  // Zone bboxes laid out so adjacent patches never overlap in plan view —
  // corners stop short of the slot, slot stops short of the net band, and
  // each label has its own keepout column for the floating chip above it.
  const zones: Zone[] = useMemo(() => [
    { id: "perim",   cx: 35, cz: 0,   w: 14, d: 66, baseColor: themeColor, label: "PERIM" },
    { id: "cornerL", cx: 78, cz: -30, w: 20, d: 18, baseColor: "#38bdf8",  label: "L CORNER" },
    { id: "cornerR", cx: 78, cz:  30, w: 20, d: 18, baseColor: "#38bdf8",  label: "R CORNER" },
    { id: "slot",    cx: 69, cz: 0,   w: 20, d: 22, baseColor: themeColor, label: "SLOT" },
    { id: "net",     cx: 85, cz: 0,   w: 6,  d: 10, baseColor: "#fbbf24",  label: "NET" },
  ], [themeColor]);

  const hoveredZone = hover != null ? zones.find(z => z.id === hover) : null;
  const hoveredVal = hover != null ? activations[hover] : null;

  return (
    <div className="relative w-full" style={{ aspectRatio: "200 / 110" }}>
      <Canvas
        frameloop={autoRotate ? "always" : "demand"}
        dpr={dpr}
        camera={{ position: [60, 60, 95], fov: 38, near: 1, far: 800 }}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        style={{ background: "transparent", touchAction: "none", cursor: freePan ? "move" : "grab" }}
      >
        <ambientLight intensity={0.55} color="#9bbcff" />
        <directionalLight position={[40, 80, 30]} intensity={0.75} color="#e0eaff" castShadow />
        <directionalLight position={[-30, 50, -20]} intensity={0.35} color="#80a8d8" />
        <hemisphereLight args={[0xbcd8ff, 0x0a1018, 0.35]} />

        <Boards themeColor={themeColor} />
        <IceSurface />
        <RinkMarkings />
        <Net x={-89} />
        <Net x={89} />

        {zones.map((z) => (
          <ZonePatch
            key={z.id}
            zone={z}
            pct={activations[z.id] ?? 0}
            hovered={hover === z.id}
            onHover={() => setHover(z.id)}
            onLeave={() => setHover(null)}
          />
        ))}

        <OrbitControls
          // @ts-expect-error drei typing — accept assignable to React ref
          ref={controlsRef}
          target={[60, 0, 0]}
          enableDamping
          dampingFactor={0.08}
          enablePan={freePan}
          enableZoom
          zoomSpeed={0.6}
          minDistance={40}
          maxDistance={220}
          minPolarAngle={0.05}
          maxPolarAngle={Math.PI / 2.02}
          autoRotate={autoRotate}
          autoRotateSpeed={-2.2}
        />
      </Canvas>

      <div className="absolute top-2 left-2 flex items-center gap-2 pointer-events-none">
        <span className="hud-mono text-[9px] uppercase tracking-[0.18em] px-2 py-0.5 rounded"
          style={{ color: themeColor, background: "rgba(0,0,0,0.55)", backdropFilter: "blur(4px)" }}>
          ◢ ZONE TENDENCY · 3D
        </span>
      </div>

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

      {hoveredZone && hoveredVal != null && (
        <div className="absolute bottom-2 right-2 hud-mono text-[10px] uppercase tracking-[0.18em] px-2 py-1 rounded border pointer-events-none"
          style={{
            color: themeColor,
            borderColor: `${themeColor}55`,
            background: "rgba(0,0,0,0.65)",
            backdropFilter: "blur(6px)",
          }}>
          ▸ {hoveredZone.label} · {hoveredVal.toFixed(1)}%
        </div>
      )}

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
              <span className="inline-block rounded-sm shrink-0" style={{ width: 8, height: 8, background: themeColor, opacity: 0.5 }} />
              <span>slot · perim · base</span>
            </div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="inline-block rounded-sm shrink-0" style={{ width: 8, height: 8, background: "#38bdf8", opacity: 0.5 }} />
              <span>corner · cycle zones</span>
            </div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="inline-block rounded-sm shrink-0" style={{ width: 8, height: 8, background: "#fbbf24", opacity: 0.65 }} />
              <span>net-front · &gt;15% activity</span>
            </div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="inline-block rounded-sm shrink-0" style={{ width: 8, height: 8, background: "#f87171", opacity: 0.75 }} />
              <span>hot · &gt;25% activity</span>
            </div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="hud-mono shrink-0" style={{ color: themeColor }}>%</span>
              <span>floating · share of OZ touches</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
