"""Frontier-learning iteration: resolve the remaining 11/26 GarmentCode
body-measurement fields using geometric means over sinew-mocap/solve's own
tools -- the ANNY FK bone skeleton (exact joint positions, not guesses)
plus the real flip_geodesics geodesic solver, the same two real tools
already used for the 15 settled fields.

iFire: 'use frontier learning and resolve every red measurement use
geometric means in the tools of https://github.com/sinew-mocap/solve'

Frontier-learning protocol per field: state the geometric definition
(bone landmark or ring-derived point), compute it, run a PLAUSIBILITY
GATE before accepting the number (never print a value that fails its
gate as if it passed), and log the attempt -- kept or rejected -- so the
iteration record is visible, not just the final numbers. See this
script's printed log and CHANGELOG.md for the kept/rejected history.
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
M_TO_CM = 100.0

model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref", phenotypes="all")
model.eval()
with torch.no_grad():
    rest_out = model(phenotype_kwargs={"gender": 1.0})
rest_bone_poses = rest_out["rest_bone_poses"]
verts = rest_out["rest_vertices"][0].numpy().astype(np.float64)
faces = np.asarray(model.faces, dtype=np.int64)
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
mesh.export(str(OBJ_PATH))

# Real ANNY bone skeleton indices (from sinew-mocap/solve's own FK rig,
# model.bone_labels -- confirmed by listing them directly, not guessed).
BONE = {
    "Hips": 1, "Spine1": 2, "Spine2": 3, "Chest": 4, "Neck1": 5, "Neck2": 6,
    "Head": 7, "HeadEnd": 8, "Jaw": 9,
    "LeftShoulder": 12, "LeftArm": 13, "LeftForeArm": 14, "LeftHand": 15,
    "RightShoulder": 40, "RightArm": 41,
    "LeftLeg": 68, "LeftShin": 69, "RightLeg": 73, "RightShin": 74,
}


def bone_pos(name):
    return rest_bone_poses[0, BONE[name], :3, 3].numpy()


def nearest_vertex(p):
    d2 = np.sum((mesh.vertices - np.asarray(p)) ** 2, axis=1)
    return int(np.argmin(d2))


def section_loop(origin, normal, centerline_point):
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None or len(sec.discrete) == 0:
        return None
    loops = sec.discrete
    centroids = [loop.mean(axis=0) for loop in loops]
    dists = [np.linalg.norm(c - centerline_point) for c in centroids]
    return loops[int(np.argmin(dists))]


def _parse_kv(stdout):
    out = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def measure_path(va, vb):
    r = subprocess.run(
        [FLIP_GEODESICS, str(OBJ_PATH), f"--measure-path={va},{vb}", "--units-scale=100"],
        capture_output=True, text=True, timeout=30,
    )
    return _parse_kv(r.stdout)


# --- real torso cross-section heights (same derivation as every prior ---
# --- script in this corpus -- recomputed here so this script is self- ---
# --- contained and cannot silently drift from a hardcoded copy). --------
HIPS_BONE_Z = bone_pos("Hips")[2]
CHEST_BONE_Z = bone_pos("Chest")[2]
z_scan = np.linspace(HIPS_BONE_Z - 0.05, CHEST_BONE_Z + 0.05, 120)
profile = np.array([
    (lambda lp: np.linalg.norm(np.diff(lp, axis=0), axis=1).sum() if lp is not None else np.nan)(
        section_loop([0, 0, z], [0, 0, 1], [0, 0, z])
    ) for z in z_scan
])
valid = ~np.isnan(profile)
waist_idx = np.nanargmin(profile)
waist_z = z_scan[waist_idx]
below_valid = np.where(z_scan < waist_z)[0]
below_valid = below_valid[valid[below_valid]]
hips_idx = below_valid[np.nanargmax(profile[below_valid])]
above_valid = np.where(z_scan > waist_z)[0]
above_valid = above_valid[valid[above_valid]]
bust_idx = above_valid[np.nanargmax(profile[above_valid])]
heights = {"waist": z_scan[waist_idx], "hips": z_scan[hips_idx], "bust": z_scan[bust_idx]}


def ring_at(z):
    return section_loop([0, 0, z], [0, 0, 1], [0, 0, z])


def leftmost_rightmost(loop_pts):
    """The two surface points of a ring farthest apart ALONG X -- the
    geometric stand-in for 'the two bust/bum points' or 'left/right of a
    width measurement' when no named landmark exists yet."""
    i_left = int(np.argmin(loop_pts[:, 0]))
    i_right = int(np.argmax(loop_pts[:, 0]))
    return loop_pts[i_left], loop_pts[i_right]


def back_side_leftmost_rightmost(loop_pts):
    """Restrict to the posterior half of the ring (y below the ring's own
    mean y -- the back surface, not the sides) before taking left/right
    extremes: a straight left-right extreme over the WHOLE ring picks the
    side points (the widest part of the torso), not the back-specific
    width back_width/*_back_width need."""
    y_mean = loop_pts[:, 1].mean()
    back = loop_pts[loop_pts[:, 1] < y_mean]
    if len(back) < 3:
        back = loop_pts
    i_left = int(np.argmin(back[:, 0]))
    i_right = int(np.argmax(back[:, 0]))
    return back[i_left], back[i_right]


log = []


def attempt(name, definition, va_pos, vb_pos, plausible_fn):
    va, vb = nearest_vertex(va_pos), nearest_vertex(vb_pos)
    out = measure_path(va, vb)
    dij = out.get("DIJKSTRA_LENGTH")
    geo = out.get("GEODESIC_LENGTH", dij)
    straight_cm = np.linalg.norm(np.asarray(va_pos) - np.asarray(vb_pos)) * M_TO_CM
    ok = geo is not None and plausible_fn(geo, straight_cm)
    entry = {"field": name, "definition": definition, "v_a": va, "v_b": vb,
              "straight_line_cm": round(float(straight_cm), 2),
              "geodesic_cm": round(geo, 2) if geo is not None else None,
              "kept": bool(ok)}
    log.append(entry)
    print(("KEPT " if ok else "REJECTED ") + f"{name}: {entry}")
    return entry if ok else None


def attempt_axis_extent(name, definition, z_a, z_b, plausible_fn):
    val_cm = abs(z_a - z_b) * M_TO_CM
    ok = plausible_fn(val_cm)
    entry = {"field": name, "definition": definition, "value_cm": round(val_cm, 2), "kept": bool(ok)}
    log.append(entry)
    print(("KEPT " if ok else "REJECTED ") + f"{name}: {entry}")
    return entry if ok else None


results = {}

# 1. shoulder_w: surface distance LeftArm <-> RightArm bone joints (the
# acromion/shoulder-ball joints -- NOT LeftShoulder/RightShoulder, which
# sit at the clavicle root near the spine: a first attempt using those
# measured only 2.89cm straight-line apart, an anatomically impossible
# "shoulder width", and was REJECTED by the plausibility gate below
# before being trusted. This is the frontier-learning iteration: the
# rejected attempt is kept in the log, not silently discarded.)
r = attempt("shoulder_w", "surface path, LeftArm bone vertex -> RightArm bone vertex "
            "(the shoulder-ball joints; LeftShoulder/RightShoulder bones were tried "
            "first and rejected -- see CHANGELOG.md)",
            bone_pos("LeftArm"), bone_pos("RightArm"),
            lambda geo, straight: 20 < straight < 60 and geo >= straight * 0.98 and geo < 80)
if r:
    results["shoulder_w"] = r

# 2. neck_w: surface distance between the neck ring's left/right-most points.
neck_z = (bone_pos("Neck1")[2] + bone_pos("Neck2")[2]) / 2
neck_loop = ring_at(neck_z)
if neck_loop is not None:
    nl, nr = leftmost_rightmost(neck_loop)
    r = attempt("neck_w", f"surface path across the neck ring at z={neck_z:.4f} "
                f"(Neck1/Neck2 bone height), left-most <-> right-most ring point",
                nl, nr, lambda geo, straight: 0 < straight < 30 and geo >= straight * 0.98 and geo < 40)
    if r:
        results["neck_w"] = r

# 3-5. {waist,hip,back}_back_width: back-half left/right extreme of each
# torso ring, at bust/waist/hips height respectively.
for field, z in [("waist_back_width", heights["waist"]),
                  ("hip_back_width", heights["hips"]),
                  ("back_width", heights["bust"])]:
    loop = ring_at(z)
    if loop is None:
        continue
    bl, br = back_side_leftmost_rightmost(loop)
    r = attempt(field, f"surface path across the BACK half (y < ring mean y) of the "
                f"{'waist' if 'waist' in field else 'hips' if 'hip' in field else 'bust'} "
                f"ring at z={z:.4f}, left-most <-> right-most back-side point",
                bl, br, lambda geo, straight, zz=z: 0 < straight < 60 and geo >= straight * 0.98 and geo < 70)
    if r:
        results[field] = r

# 6. head_l: axis extent, HeadEnd (crown) to Jaw (chin).
r = attempt_axis_extent("head_l", "axis extent, HeadEnd bone z - Jaw bone z (crown to chin)",
                         bone_pos("HeadEnd")[2], bone_pos("Jaw")[2],
                         lambda v: 10 < v < 30)
if r:
    results["head_l"] = r

# 7. armscye_depth: surface path, LeftShoulder bone vertex down to the
# bust ring's left-most (under-arm-side) point -- the vertical drop from
# shoulder to underarm level a sleeve's armhole actually spans.
bust_loop = ring_at(heights["bust"])
if bust_loop is not None:
    bl_bust, _ = leftmost_rightmost(bust_loop)
    r = attempt("armscye_depth", "surface path, LeftShoulder bone vertex -> bust ring's "
                "left-most (underarm-side) point",
                bone_pos("LeftShoulder"), bl_bust,
                lambda geo, straight: 0 < straight < 35 and geo >= straight * 0.98 and geo < 45)
    if r:
        results["armscye_depth"] = r

# 8. vert_bust_line: axis extent, Neck1/Neck2 midpoint z - bust ring z.
r = attempt_axis_extent("vert_bust_line", "axis extent, Neck1/Neck2 midpoint z - bust ring z",
                         neck_z, heights["bust"], lambda v: 5 < v < 40)
if r:
    results["vert_bust_line"] = r

# 9. waist_over_bust_line: axis extent, bust ring z - waist ring z (already
# have both exact heights from the torso scan above -- trivial once the
# circumference solve has run, not a new measurement technique).
r = attempt_axis_extent("waist_over_bust_line", "axis extent, bust ring z - waist ring z",
                         heights["bust"], heights["waist"], lambda v: 2 < v < 30)
if r:
    results["waist_over_bust_line"] = r

# 10. bust_points: surface distance between the bust ring's left-most and
# right-most points (the geometric stand-in for the two bust apexes).
if bust_loop is not None:
    bl, br = leftmost_rightmost(bust_loop)
    r = attempt("bust_points", "surface path between the bust ring's left-most and "
                "right-most points", bl, br,
                lambda geo, straight: 0 < straight < 40 and geo >= straight * 0.98 and geo < 50)
    if r:
        results["bust_points"] = r

# 11. bum_points: same idea at the hips ring.
hips_loop = ring_at(heights["hips"])
if hips_loop is not None:
    hl, hr = leftmost_rightmost(hips_loop)
    r = attempt("bum_points", "surface path between the hips ring's left-most and "
                "right-most points", hl, hr,
                lambda geo, straight: 0 < straight < 50 and geo >= straight * 0.98 and geo < 60)
    if r:
        results["bum_points"] = r

print()
print(f"Resolved {len(results)} / 11 remaining fields with a kept measurement.")
rejected = [e for e in log if not e["kept"]]
print(f"Rejected (plausibility gate failed): {[e['field'] for e in rejected]}")

with open(OUT_DIR / "frontier_remaining_fields_log.json", "w") as f:
    json.dump({"kept": results, "full_log": log}, f, indent=2)
print(json.dumps(results, indent=2))
