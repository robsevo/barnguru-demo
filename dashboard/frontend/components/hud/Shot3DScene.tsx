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
  /** Render a goalie figure inside each crease and improve the net mesh.
   *  Use on goalie shot-against maps. */
  goalie?: boolean;
  /** Per-`shot.type` color override. Shots fall back to themeColor when
   *  their type isn't in the map (or when the map is omitted). */
  colorByType?: Record<string, string>;
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
function RinkMarkings({ hideGoalLines = false }: { hideGoalLines?: boolean }) {
  return (
    <group position={[0, 0.02, 0]}>
      {/* Center red line */}
      <RinkLine x1={0} z1={-RINK_WID / 2 + 2} x2={0} z2={RINK_WID / 2 - 2} color="#e63a3a" width={0.5} />
      {/* Blue lines */}
      <RinkLine x1={-25} z1={-RINK_WID / 2 + 2} x2={-25} z2={RINK_WID / 2 - 2} color="#3b8fd0" width={0.5} />
      <RinkLine x1={25}  z1={-RINK_WID / 2 + 2} x2={25}  z2={RINK_WID / 2 - 2} color="#3b8fd0" width={0.5} />
      {/* Goal lines (red) — hidden on goalie views so the net stops reading as a square cut by a line */}
      {!hideGoalLines && (
        <>
          <RinkLine x1={-89} z1={-RINK_WID / 2 + 8} x2={-89} z2={RINK_WID / 2 - 8} color="#e63a3a" width={0.3} />
          <RinkLine x1={89}  z1={-RINK_WID / 2 + 8} x2={89}  z2={RINK_WID / 2 - 8} color="#e63a3a" width={0.3} />
        </>
      )}

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

// ─── 3D net — frame + mesh netting ─────────────────────────────────────────
function Net({ x }: { x: number }) {
  const sign = x < 0 ? -1 : 1;
  const netDepth = 3.6;
  const netWidth = 6;
  const netHeight = 4;
  // Real hockey nets taper inward at the back. Front mouth width 6ft,
  // back panel width ~3.6ft sitting roughly 3.6ft behind the goal line.
  const backHalfWidth = 1.8;

  // Build the four "skirt" net panels (left, right, top, bottom) plus the
  // back panel using thin tubes to create the woven look without the cost
  // of a real geometry shader.
  const meshLines = useMemo(() => {
    const lines: { a: [number, number, number]; b: [number, number, number] }[] = [];
    const fwdX = 0;             // local x of front mouth (group is anchored at goal line)
    const backX = sign * netDepth;
    const fzL = -netWidth / 2;
    const fzR =  netWidth / 2;
    const bzL = -backHalfWidth;
    const bzR =  backHalfWidth;
    const yT = netHeight;
    const yB = 0.05;            // hover above ice

    // Top panel — front edge to back edge
    for (let i = 0; i <= 6; i++) {
      const t = i / 6;
      const fz = fzL + (fzR - fzL) * t;
      const bz = bzL + (bzR - bzL) * t;
      lines.push({ a: [fwdX, yT, fz], b: [backX, yT, bz] });
    }
    // Top panel — horizontal weft
    for (let i = 1; i < 4; i++) {
      const t = i / 4;
      const fx = fwdX + (backX - fwdX) * t;
      const zL = fzL + (bzL - fzL) * t;
      const zR = fzR + (bzR - fzR) * t;
      lines.push({ a: [fx, yT, zL], b: [fx, yT, zR] });
    }
    // Left side panel (vertical strands)
    for (let i = 0; i <= 6; i++) {
      const t = i / 6;
      const fx = fwdX + (backX - fwdX) * t;
      const fz = fzL + (bzL - fzL) * t;
      lines.push({ a: [fx, yB, fz], b: [fx, yT, fz] });
    }
    // Right side panel (vertical strands)
    for (let i = 0; i <= 6; i++) {
      const t = i / 6;
      const fx = fwdX + (backX - fwdX) * t;
      const fz = fzR + (bzR - fzR) * t;
      lines.push({ a: [fx, yB, fz], b: [fx, yT, fz] });
    }
    // Back panel (vertical strands)
    for (let i = 0; i <= 4; i++) {
      const t = i / 4;
      const z = bzL + (bzR - bzL) * t;
      lines.push({ a: [backX, yB, z], b: [backX, yT, z] });
    }
    // Back panel (horizontal strands)
    for (let i = 1; i < 4; i++) {
      const y = (i / 4) * yT;
      lines.push({ a: [backX, y, bzL], b: [backX, y, bzR] });
    }
    return lines;
  }, [sign]);

  const meshPositions = useMemo(() => {
    const arr: number[] = [];
    for (const l of meshLines) {
      arr.push(...l.a, ...l.b);
    }
    return new Float32Array(arr);
  }, [meshLines]);

  return (
    <group position={[x, 0, 0]}>
      {/* Crossbar (top, horizontal across goal mouth) */}
      <mesh position={[0, netHeight, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.18, 0.18, netWidth, 8]} />
        <meshStandardMaterial color="#ff3030" emissive="#ff3030" emissiveIntensity={0.45} />
      </mesh>
      {/* Two side posts (vertical) */}
      <mesh position={[0, netHeight / 2, -netWidth / 2]}>
        <cylinderGeometry args={[0.18, 0.18, netHeight, 8]} />
        <meshStandardMaterial color="#ff3030" emissive="#ff3030" emissiveIntensity={0.45} />
      </mesh>
      <mesh position={[0, netHeight / 2, netWidth / 2]}>
        <cylinderGeometry args={[0.18, 0.18, netHeight, 8]} />
        <meshStandardMaterial color="#ff3030" emissive="#ff3030" emissiveIntensity={0.45} />
      </mesh>

      {/* Back frame — thin tubular ring along the back edge of the netting */}
      <mesh position={[sign * netDepth, netHeight, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.10, 0.10, backHalfWidth * 2, 6]} />
        <meshStandardMaterial color="#aa1f1f" emissive="#aa1f1f" emissiveIntensity={0.30} />
      </mesh>

      {/* Net mesh — dense line lattice forming the four panels */}
      <line>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[meshPositions, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#e8efff" transparent opacity={0.34} />
      </line>

      {/* Faint white wash inside the net so it reads as enclosed volume */}
      <mesh position={[sign * netDepth * 0.45, netHeight / 2, 0]}>
        <boxGeometry args={[netDepth * 0.9, netHeight - 0.2, netWidth - 0.3]} />
        <meshBasicMaterial color="#cce0ff" transparent opacity={0.04} />
      </mesh>
    </group>
  );
}

// ─── Goalie figure — minimal silhouette in the crease ───────────────────────
function GoalieFigure({ x, themeColor }: { x: number; themeColor: string }) {
  const sign = x < 0 ? -1 : 1;
  // 4ft in front of the goal line, facing inward toward the rink centre
  const cx = x - sign * 4;
  const padColor = "#dde3eb";
  const bodyColor = themeColor;
  return (
    <group position={[cx, 0, 0]}>
      {/* Pads — two upright cylinders for the legs */}
      <mesh position={[0, 1.4, -0.85]}>
        <cylinderGeometry args={[0.55, 0.55, 2.8, 10]} />
        <meshStandardMaterial color={padColor} emissive={padColor} emissiveIntensity={0.08} roughness={0.6} />
      </mesh>
      <mesh position={[0, 1.4, 0.85]}>
        <cylinderGeometry args={[0.55, 0.55, 2.8, 10]} />
        <meshStandardMaterial color={padColor} emissive={padColor} emissiveIntensity={0.08} roughness={0.6} />
      </mesh>
      {/* Body/chest protector */}
      <mesh position={[0, 3.7, 0]}>
        <cylinderGeometry args={[0.95, 1.10, 2.3, 12]} />
        <meshStandardMaterial color={bodyColor} emissive={bodyColor} emissiveIntensity={0.18} roughness={0.55} />
      </mesh>
      {/* Glove — extended toward the shooter (rink centre) */}
      <mesh position={[-sign * 1.5, 3.2, -1.4]}>
        <sphereGeometry args={[0.55, 10, 10]} />
        <meshStandardMaterial color={bodyColor} emissive={bodyColor} emissiveIntensity={0.20} roughness={0.5} />
      </mesh>
      {/* Blocker */}
      <mesh position={[-sign * 1.2, 3.2, 1.4]}>
        <boxGeometry args={[0.7, 0.8, 0.9]} />
        <meshStandardMaterial color={bodyColor} emissive={bodyColor} emissiveIntensity={0.20} roughness={0.5} />
      </mesh>
      {/* Helmet */}
      <mesh position={[0, 5.4, 0]}>
        <sphereGeometry args={[0.62, 14, 14]} />
        <meshStandardMaterial color="#101418" emissive="#1a1f25" emissiveIntensity={0.10} roughness={0.4} />
      </mesh>
      {/* Helmet cage hint — small ring on the forward face */}
      <mesh position={[-sign * 0.5, 5.4, 0]} rotation={[0, 0, sign * Math.PI / 2]}>
        <torusGeometry args={[0.35, 0.04, 6, 16]} />
        <meshStandardMaterial color="#5a6470" emissive="#5a6470" emissiveIntensity={0.20} />
      </mesh>
      {/* Stick blade — flat on the ice between the pads */}
      <mesh position={[-sign * 1.6, 0.1, 0]} rotation={[0, 0, 0]}>
        <boxGeometry args={[1.6, 0.05, 0.45]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
    </group>
  );
}

// ─── Goal trajectory arcs — subtle curved lines from shot → net ────────────
function GoalTrajectories({ shots, themeColor, flip, colorByType }: { shots: Shot[]; themeColor: string; flip: boolean; colorByType?: Record<string, string> }) {
  const goals = useMemo(() => shots.filter((s) => s.goal), [shots]);
  const fx = (v: number) => (flip ? -v : v);

  const arcs = useMemo(() => {
    return goals.map((s) => {
      const sx = fx(s.x);
      const sz = s.y;
      // Aim at the goal line on whichever side the shot came from
      const goalX = sx >= 0 ? 89 : -89;
      const start = new THREE.Vector3(sx, 0.9, sz);
      const end   = new THREE.Vector3(goalX, 1.6, 0);
      // Lift the middle so the arc reads as a trajectory above the ice
      const midY  = 4 + Math.min(2.5, Math.abs(goalX - sx) * 0.04);
      const mid   = new THREE.Vector3((sx + goalX) / 2, midY, sz * 0.4);
      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      const points = curve.getPoints(18);
      const arr = new Float32Array(points.length * 3);
      for (let i = 0; i < points.length; i++) {
        arr[i * 3 + 0] = points[i].x;
        arr[i * 3 + 1] = points[i].y;
        arr[i * 3 + 2] = points[i].z;
      }
      // Arrow head direction (last segment)
      const headDir = points[points.length - 1].clone().sub(points[points.length - 2]).normalize();
      const color = (s.type && colorByType?.[s.type]) ?? themeColor;
      return { positions: arr, end, headDir, color };
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [goals, flip, colorByType, themeColor]);

  if (arcs.length === 0) return null;

  return (
    <group>
      {arcs.map((a, i) => (
        <group key={i}>
          <line>
            <bufferGeometry>
              <bufferAttribute attach="attributes-position" args={[a.positions, 3]} />
            </bufferGeometry>
            <lineBasicMaterial color={a.color} transparent opacity={0.42} />
          </line>
          {/* Arrowhead — tiny cone pointing along headDir */}
          <mesh
            position={[a.end.x, a.end.y, a.end.z]}
            quaternion={(() => {
              // Cone default points along +y; rotate to point along headDir.
              const q = new THREE.Quaternion();
              q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), a.headDir);
              return q;
            })()}
          >
            <coneGeometry args={[0.45, 1.2, 8]} />
            <meshStandardMaterial color={a.color} emissive={a.color} emissiveIntensity={0.55} transparent opacity={0.65} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

// ─── Shot instancing ───────────────────────────────────────────────────────
// Group by type so per-team coloring works with the GPU-instanced renderer
// (one material per group → one color per group). With colorByType absent
// or empty everything falls into a single "_" group at themeColor — same
// path as the old single-color renderer.
function ShotInstances({ shots, themeColor, flip, colorByType }: { shots: Shot[]; themeColor: string; flip: boolean; colorByType?: Record<string, string> }) {
  const fx = (s: Shot) => (flip ? -s.x : s.x);

  const groups = useMemo(() => {
    const map = new Map<string, { color: string; misses: Shot[]; goals: Shot[] }>();
    for (const s of shots) {
      const key = s.type ?? "_";
      const color = (s.type && colorByType?.[s.type]) ?? themeColor;
      let g = map.get(key);
      if (!g) { g = { color, misses: [], goals: [] }; map.set(key, g); }
      (s.goal ? g.goals : g.misses).push(s);
    }
    return Array.from(map.entries());
  }, [shots, colorByType, themeColor]);

  return (
    <group>
      {groups.map(([key, g]) => (
        <group key={key}>
          {g.misses.length > 0 ? (
            <Instances limit={Math.max(1, g.misses.length)} range={g.misses.length}>
              <sphereGeometry args={[1.0, 12, 12]} />
              <meshStandardMaterial color={g.color} emissive={g.color} emissiveIntensity={0.40} roughness={0.4} />
              {g.misses.map((s, i) => (
                <Instance key={i} position={[fx(s), 0.8, s.y]} scale={0.9} />
              ))}
            </Instances>
          ) : null}
          {g.goals.length > 0 ? (
            <Instances limit={Math.max(1, g.goals.length)} range={g.goals.length}>
              <sphereGeometry args={[1.3, 14, 14]} />
              <meshStandardMaterial
                color="#ffffff"
                emissive={g.color}
                emissiveIntensity={1.4}
                metalness={0.4}
                roughness={0.2}
              />
              {g.goals.map((s, i) => (
                <Instance key={i} position={[fx(s), 1.0, s.y]} scale={1.2} />
              ))}
            </Instances>
          ) : null}
        </group>
      ))}
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

export default function Shot3DScene({ shots, themeColor = "#C9A84C", flip = false, goalie = false, colorByType }: Shot3DSceneProps) {
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
        <RinkMarkings hideGoalLines={goalie} />
        <Net x={-89} />
        <Net x={89} />
        {goalie && (
          <>
            <GoalieFigure x={-89} themeColor={themeColor} />
            <GoalieFigure x={89}  themeColor={themeColor} />
          </>
        )}
        <ShotInstances shots={shots} themeColor={themeColor} flip={flip} colorByType={colorByType} />
        <GoalTrajectories shots={shots} themeColor={themeColor} flip={flip} colorByType={colorByType} />

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
            {colorByType && Object.keys(colorByType).length > 0 ? (
              // Two-team game: list each team with its color so the rink reads
              // unambiguously instead of leaning on a single themeColor swatch.
              Object.entries(colorByType).map(([team, color]) => (
                <div key={team} className="flex items-center gap-1.5 mt-0.5 first:mt-0">
                  <span className="inline-block rounded-full shrink-0" style={{ width: 8, height: 8, background: color, boxShadow: `0 0 6px ${color}` }} />
                  <span>{team} · shot</span>
                </div>
              ))
            ) : (
              <>
                <div className="flex items-center gap-1.5">
                  <span className="inline-block rounded-full shrink-0" style={{ width: 8, height: 8, background: themeColor, boxShadow: `0 0 6px ${themeColor}` }} />
                  <span>shot · saved</span>
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="inline-block rounded-full shrink-0"
                    style={{ width: 10, height: 10, background: "#fff", boxShadow: `0 0 10px ${themeColor}, inset 0 0 4px ${themeColor}` }} />
                  <span>goal · scored</span>
                </div>
              </>
            )}
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="inline-block rounded-full shrink-0"
                style={{ width: 10, height: 10, background: "#fff", boxShadow: "0 0 10px rgba(255,255,255,0.85), inset 0 0 4px rgba(255,255,255,0.7)" }} />
              <span>goal · scored (white core)</span>
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
