import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import type { Mesh, Points } from "three";
import { AdditiveBlending, BufferAttribute, BufferGeometry } from "three";

import type { BatchImageClassificationTask } from "./types";

type GalaxyCanvasProps = {
  task: BatchImageClassificationTask;
};

export function GalaxyCanvas({ task }: GalaxyCanvasProps) {
  return (
    <section className="absolute inset-0" aria-label="Galaxy AI 星系任务画布">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_32%_45%,rgba(34,211,238,0.18),transparent_22%),radial-gradient(circle_at_13%_32%,rgba(124,58,237,0.28),transparent_28%),radial-gradient(circle_at_68%_85%,rgba(168,85,247,0.16),transparent_30%),linear-gradient(120deg,#03060d_0%,#050816_48%,#02040a_100%)]" />
      <div className="absolute inset-0 opacity-35 [background-image:radial-gradient(circle_at_center,rgba(255,255,255,0.95)_0_1px,transparent_1px)] [background-size:58px_42px]" />
      <Canvas
        camera={{ position: [0, 0, 8], fov: 50 }}
        className="absolute inset-0 z-10"
        dpr={[1, 1.6]}
      >
        <ambientLight intensity={0.5} />
        <pointLight position={[-4, 3, 4]} intensity={1.6} color="#67e8f9" />
        <pointLight position={[3, -2, 3]} intensity={1.1} color="#a78bfa" />
        <StarCloud />
        <OrbitRings />
        <NebulaPlanet />
      </Canvas>

      <div className="galaxy-visual-fallback" aria-hidden="true">
        <div className="orbit orbit-one" />
        <div className="orbit orbit-two" />
        <div className="orbit orbit-three" />
        <div className="nebula-planet" />
      </div>

      <div className="pointer-events-none absolute left-[36%] top-[61%] z-20 hidden -translate-x-1/2 rounded-lg border border-cyan-300/60 bg-black/55 px-6 py-4 shadow-[0_0_34px_rgba(34,211,238,0.32)] backdrop-blur-xl lg:block">
        <div className="flex items-center gap-3">
          <span className="size-3 rounded-full bg-emerald-400 shadow-[0_0_18px_rgba(52,211,153,0.9)]" />
          <div>
            <p className="text-2xl font-semibold text-white">{task.name}</p>
            <p className="mt-1 text-sm text-slate-300">{task.task_type}</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function StarCloud() {
  const pointsRef = useRef<Points>(null);
  const geometry = useMemo(() => {
    const vertices: number[] = [];
    for (let index = 0; index < 900; index += 1) {
      vertices.push(
        (Math.random() - 0.5) * 14,
        (Math.random() - 0.5) * 9,
        (Math.random() - 0.5) * 4,
      );
    }

    const starGeometry = new BufferGeometry();
    starGeometry.setAttribute(
      "position",
      new BufferAttribute(new Float32Array(vertices), 3),
    );
    return starGeometry;
  }, []);

  useFrame((_, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.z += delta * 0.006;
    }
  });

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        color="#dbeafe"
        size={0.016}
        transparent
        opacity={0.9}
        blending={AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

function OrbitRings() {
  return (
    <group rotation={[0.95, 0.08, -0.08]} position={[-1.8, -0.15, -0.4]}>
      {[2.1, 3.05, 4.0].map((radius, index) => (
        <mesh key={radius}>
          <ringGeometry args={[radius, radius + 0.008, 160]} />
          <meshBasicMaterial
            color={index === 2 ? "#22d3ee" : "#0e7490"}
            transparent
            opacity={index === 2 ? 0.46 : 0.22}
            side={2}
          />
        </mesh>
      ))}
    </group>
  );
}

function NebulaPlanet() {
  const planetRef = useRef<Mesh>(null);
  const haloRef = useRef<Mesh>(null);

  useFrame((_, delta) => {
    if (planetRef.current) {
      planetRef.current.rotation.y += delta * 0.18;
      planetRef.current.rotation.z += delta * 0.025;
    }
    if (haloRef.current) {
      haloRef.current.rotation.z -= delta * 0.08;
    }
  });

  return (
    <group position={[-2.1, -0.15, 0]}>
      <mesh ref={haloRef}>
        <ringGeometry args={[1.95, 2.12, 160]} />
        <meshBasicMaterial
          color="#22d3ee"
          transparent
          opacity={0.38}
          blending={AdditiveBlending}
          side={2}
        />
      </mesh>
      <mesh ref={planetRef}>
        <sphereGeometry args={[1.45, 80, 80]} />
        <meshStandardMaterial
          color="#252bff"
          emissive="#28117c"
          emissiveIntensity={0.8}
          roughness={0.35}
          metalness={0.15}
        />
      </mesh>
      <mesh scale={[1.47, 1.47, 1.47]}>
        <sphereGeometry args={[1.45, 80, 80]} />
        <meshBasicMaterial
          color="#8b5cf6"
          transparent
          opacity={0.2}
          blending={AdditiveBlending}
        />
      </mesh>
    </group>
  );
}
