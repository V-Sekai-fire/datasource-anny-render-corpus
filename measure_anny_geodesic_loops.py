"""Measure the GarmentCode circumference fields as REAL geodesic loops (not
plane cross-sections) by running the actual flip_geodesics headless CLI
(V-Sekai-fire/TOOL_geodesiac_path) against the ANNY-SOMA-X mesh.

For each circumference field, picks a ring of ACTUAL MESH VERTICES near the
cross-section height/plane already found by measure_anny_circumferences.py
(one nearest vertex per angular bin around the body's centerline axis), then
calls:

    flip_geodesics anny-soma-x.obj --measure-loop v1,v2,...,vN --units-scale 100

and reads back DIJKSTRA_LENGTH (initial edge-walk length) and
DELAUNAY_LENGTH (the number to trust -- a pure retriangulation, safe for a
closed loop; see TOOL_geodesiac_path's own module comment on why
iterativeShorten must never run on a loop).
"""
import json
import pathlib
import subprocess

import numpy as np
import torch
import trimesh
from anny import Anny

FLIP_GEODESICS = "/Users/ernest.lee/.hermes/cache/scratch/geodesic-probe/build/bin/flip_geodesics"
OUT_DIR = pathlib.Path("build/anny_export")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OBJ_PATH = OUT_DIR / "anny-soma-x.obj"

model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref", phenotypes="all")
model.eval()
with torch.no_grad():
    rest_out = model(phenotype_kwargs={"gender": 1.0})
rest_bone_poses = rest_out["rest_bone_poses"]
verts = rest_out["rest_vertices"][0].numpy().astype(np.float64)
faces = np.asarray(model.faces, dtype=np.int64)
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
mesh.export(str(OBJ_PATH))
print(f"wrote {OBJ_PATH} ({len(verts)} verts, {len(faces)} faces)")

M_TO_CM = 100.0


def vertex_ring_from_section_loop(mesh, origin, normal, centerline_point):
    """Snap the ALREADY-VALIDATED cross-section loop (the one
    `section_perimeter` in measure_anny_circumferences.py picks: the
    discrete loop whose centroid is nearest `centerline_point`, which is
    what correctly discards the arm loops crossing the torso band) to
    actual mesh vertex indices, one nearest vertex per loop point, in loop
    order. This is the fix for a naive angular-bin-over-all-near-plane-
    vertices picker, which silently pulled in arm vertices that also cross
    the band and produced a 276cm "waist" -- nonsense caught by comparing
    against the already-validated plane-cross-section perimeter (62.49cm)
    before trusting this ring at all."""
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None or len(sec.discrete) == 0:
        return None, None
    loops = sec.discrete
    centroids = [loop.mean(axis=0) for loop in loops]
    dists = [np.linalg.norm(c - centerline_point) for c in centroids]
    best = int(np.argmin(dists))
    loop_pts = loops[best]
    # snap each loop point to its nearest ACTUAL mesh vertex
    ring = []
    for p in loop_pts:
        d2 = np.sum((mesh.vertices - p) ** 2, axis=1)
        vtx = int(np.argmin(d2))
        if not ring or ring[-1] != vtx:
            ring.append(vtx)
    if ring and ring[0] == ring[-1]:
        ring.pop()
    # the reference perimeter the snapped ring is being compared against
    d = np.diff(loop_pts, axis=0)
    ref_perim = np.linalg.norm(d, axis=1).sum()
    if not np.allclose(loop_pts[0], loop_pts[-1]):
        ref_perim += np.linalg.norm(loop_pts[0] - loop_pts[-1])
    return ring, ref_perim * M_TO_CM


def _parse_kv_lines(stdout):
    out = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            key, val = parts
            try:
                out[key] = float(val)
            except ValueError:
                pass
    return out


def run_measure_loop(obj_path, ring):
    if len(ring) < 3:
        return None
    arg = ",".join(str(v) for v in ring)
    r = subprocess.run(
        [FLIP_GEODESICS, str(obj_path), f"--measure-loop={arg}", "--units-scale=100"],
        capture_output=True, text=True, timeout=30,
    )
    out = _parse_kv_lines(r.stdout)
    return out if out else {"ERROR": r.stdout + r.stderr, "returncode": r.returncode}


def run_measure_path(obj_path, a, b):
    r = subprocess.run(
        [FLIP_GEODESICS, str(obj_path), f"--measure-path={a},{b}", "--units-scale=100"],
        capture_output=True, text=True, timeout=30,
    )
    out = _parse_kv_lines(r.stdout)
    return out if out else {"ERROR": r.stdout + r.stderr, "returncode": r.returncode}


# Torso circumference heights, taken from the already-validated plane-
# cross-section script's own output (waist=0.1873m, hips=0.0257m,
# bust=0.2966m, underbust=0.1896m on this mesh -- re-derive here so this
# script stands alone and doesn't silently drift from a hardcoded copy).
HIPS_BONE, CHEST_BONE = 1, 4
hips_z = rest_bone_poses[0, HIPS_BONE, 2, 3].item()
chest_z = rest_bone_poses[0, CHEST_BONE, 2, 3].item()


def section_perimeter(mesh, origin, normal, centerline_point):
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None or len(sec.discrete) == 0:
        return None
    loops = sec.discrete
    centroids = [loop.mean(axis=0) for loop in loops]
    dists = [np.linalg.norm(c - centerline_point) for c in centroids]
    best = int(np.argmin(dists))
    chosen = loops[best]
    d = np.diff(chosen, axis=0)
    total = np.linalg.norm(d, axis=1).sum()
    if not np.allclose(chosen[0], chosen[-1]):
        total += np.linalg.norm(chosen[0] - chosen[-1])
    return total


z_scan = np.linspace(hips_z - 0.05, chest_z + 0.05, 120)
profile = np.array([
    (section_perimeter(mesh, [0, 0, z], [0, 0, 1], [0, 0, z]) or np.nan) for z in z_scan
])
valid = ~np.isnan(profile)
waist_idx = np.nanargmin(profile)
waist_z = z_scan[waist_idx]
below = np.where(z_scan < waist_z)[0]
below_valid = below[valid[below]]
hips_idx = below_valid[np.nanargmax(profile[below_valid])]
above = np.where(z_scan > waist_z)[0]
above_valid = above[valid[above]]
bust_idx = above_valid[np.nanargmax(profile[above_valid])]
between = np.where((z_scan > waist_z) & (z_scan < z_scan[bust_idx]))[0]
between_valid = between[valid[between]]
underbust_idx = between_valid[np.nanargmin(profile[between_valid])] if len(between_valid) > 2 else waist_idx

heights = {
    "waist": z_scan[waist_idx],
    "hips": z_scan[hips_idx],
    "bust": z_scan[bust_idx],
    "underbust": z_scan[underbust_idx],
}

results = {}
for name, z in heights.items():
    ring, ref_perim_cm = vertex_ring_from_section_loop(mesh, [0, 0, z], [0, 0, 1], [0, 0, z])
    if not ring or len(ring) < 8:
        print(f"{name}: no usable loop/ring found")
        results[name] = {"error": "could not pick a usable vertex ring"}
        continue
    print(f"{name}: z={z:.4f}  ring_size={len(ring)}  ref_section_perim={ref_perim_cm:.2f}cm")
    out = run_measure_loop(OBJ_PATH, ring)
    # Sanity gate: a vertex-snapped ring must be close to the reference
    # cross-section perimeter it was snapped from, or the snap picked a
    # bad/self-crossing ring and the CLI's output must not be trusted.
    dij = out.get("DIJKSTRA_LENGTH") if isinstance(out, dict) else None
    plausible = bool(dij is not None and abs(dij - ref_perim_cm) / ref_perim_cm < 0.15)
    results[name] = {"ring_size": len(ring), "ref_section_perim_cm": round(ref_perim_cm, 2),
                      "plausible_vs_reference": plausible, **out}
    print(f"  {out}  plausible_vs_reference={plausible}")

with open(OUT_DIR / "geodesic_loop_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# --- arm_length (open path): shoulder bone -> wrist bone, snapped to ---
# --- nearest mesh vertices, via the real --measure-path. --------------
LEFT_ARM, LEFT_HAND = 13, 15  # LeftArm (shoulder), LeftHand (wrist) bones
shoulder_pos = rest_bone_poses[0, LEFT_ARM, :3, 3].numpy()
wrist_pos = rest_bone_poses[0, LEFT_HAND, :3, 3].numpy()


def nearest_vertex(mesh, p):
    d2 = np.sum((mesh.vertices - np.asarray(p)) ** 2, axis=1)
    return int(np.argmin(d2))


v_shoulder = nearest_vertex(mesh, shoulder_pos)
v_wrist = nearest_vertex(mesh, wrist_pos)
straight_line_cm = np.linalg.norm(shoulder_pos - wrist_pos) * M_TO_CM
path_out = run_measure_path(OBJ_PATH, v_shoulder, v_wrist)
dij = path_out.get("DIJKSTRA_LENGTH") if isinstance(path_out, dict) else None
# A surface geodesic between two landmarks must be >= the straight-line
# (chord) distance between them (triangle inequality) -- the plausibility
# gate for an open path, mirroring the loop gate above.
arm_plausible = bool(dij is not None and dij >= straight_line_cm * 0.98)
print(f"arm_length: v_shoulder={v_shoulder} v_wrist={v_wrist} "
      f"straight_line={straight_line_cm:.2f}cm  {path_out}  plausible={arm_plausible}")
results["arm_length"] = {"v_shoulder": v_shoulder, "v_wrist": v_wrist,
                          "straight_line_cm": round(float(straight_line_cm), 2),
                          "plausible_vs_straight_line": arm_plausible, **path_out}
with open(OUT_DIR / "geodesic_loop_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
