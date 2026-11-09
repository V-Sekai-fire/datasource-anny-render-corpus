"""Export REAL curved geodesic polylines (via the actual flip_geodesics CLI,
post-FlipOut-straightening / post-Delaunay) for every settled surface-
geodesic field, for the Godot x-ray contact-sheet render, and compute
every other field's real value with the CORRECT operation.

CORRECTED 2026-11-08 against the actual published reference
implementation (github.com/mbotsch/GarmentMeasurements, the real C++
source behind GarmentCodeData, arxiv.org/abs/2405.17609v3) -- read
directly, not re-derived from paper prose:
  - shoulder_w, armscye_depth, bust_points, bum_points: Euclidean
    straight-line distance (measure_landmark_distance_euclidean), not a
    surface geodesic.
  - neck_w: the full neck ring's edge-loop length (measure_vertex_trace),
    not a left-right chord.
  - leg_circ: plane shifted down until both legs separate; MINIMUM of
    the two legs at that one height.
  - head_l: nape-of-neck landmark to crown (measure_head_l), not jaw.
  - shoulder_incl, hip_inclination, arm_pose_angle: the real
    angle_plane_vec formula (plane-normal vs. landmark-vector angle),
    not a bone-rotation comparison.
See CHANGELOG.md for every old -> new value.
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
PATHS_DIR = OUT_DIR / "paths"
PATHS_DIR.mkdir(parents=True, exist_ok=True)
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


def section_loop(origin, normal, centerline_point):
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None or len(sec.discrete) == 0:
        return None
    loops = sec.discrete
    centroids = [loop.mean(axis=0) for loop in loops]
    dists = [np.linalg.norm(c - centerline_point) for c in centroids]
    return loops[int(np.argmin(dists))]


def ring_from_loop(loop_pts):
    ring = []
    for p in loop_pts:
        d2 = np.sum((mesh.vertices - p) ** 2, axis=1)
        vtx = int(np.argmin(d2))
        if not ring or ring[-1] != vtx:
            ring.append(vtx)
    if ring and ring[0] == ring[-1]:
        ring.pop()
    return ring


def nearest_vertex(p):
    d2 = np.sum((mesh.vertices - np.asarray(p)) ** 2, axis=1)
    return int(np.argmin(d2))


def leftmost_rightmost(loop_pts):
    i_left = int(np.argmin(loop_pts[:, 0]))
    i_right = int(np.argmax(loop_pts[:, 0]))
    return loop_pts[i_left], loop_pts[i_right]


def rightmost(loop_pts):
    return loop_pts[int(np.argmax(loop_pts[:, 0]))]


def back_side_leftmost_rightmost(loop_pts):
    y_mean = loop_pts[:, 1].mean()
    back = loop_pts[loop_pts[:, 1] < y_mean]
    if len(back) < 3:
        back = loop_pts
    i_left = int(np.argmin(back[:, 0]))
    i_right = int(np.argmax(back[:, 0]))
    return back[i_left], back[i_right]


def angle_plane_vec(plane_n, vector):
    """The EXACT published angle_plane_vec formula
    (mbotsch/GarmentMeasurements measurement_utils.cpp)."""
    v = vector / np.linalg.norm(vector)
    cos_angle = np.clip(np.dot(plane_n, v), -1.0, 1.0)
    theta = np.arccos(cos_angle) - np.pi / 2
    return float(-theta * 180.0 / np.pi)


def export_path(name, a, b):
    out_json = PATHS_DIR / f"{name}.json"
    r = subprocess.run(
        [FLIP_GEODESICS, str(OBJ_PATH), f"--measure-path={a},{b}", "--units-scale=1",
         f"--export-path={out_json}"],
        capture_output=True, text=True, timeout=30,
    )
    lines = r.stdout.strip().splitlines()
    print(name, "path", a, b, "->", lines[:2])
    geo_len = None
    for line in lines:
        if line.startswith("GEODESIC_LENGTH"):
            geo_len = float(line.split()[1]) * M_TO_CM
    return str(out_json), geo_len


def export_loop(name, ring):
    """Real closed geodesic loop. Uses --measure-loop-adaptive-shorten
    (geodesic-probe, added 2026-11-08: iFire "please use the tool for
    geodesic paths pls and fix it" -- the raw piecewise-Dijkstra
    construction has sharp per-vertex kinks, up to 60deg turning angle,
    genuinely present in 3D, not a 2D-projection artifact; verified by
    computing turning angles along the exported polyline directly).

    An earlier version of this function used a FIXED
    --measure-loop-bounded-shorten budget (200 iterations, 10% max
    length decrease) empirically tuned against THIS ONE mesh's loop
    sizes. iFire: "it has to work for possible anny and custom models
    so you can't hard code" -- correct: that fixed budget has no
    principled basis for a different ANNY phenotype or a wholly custom
    model (a smaller/coarser mesh could still be jagged at that
    iteration count, or could already be sliding toward collapse
    before reaching it). --measure-loop-adaptive-shorten instead stops
    on a real geometric criterion (the loop's max turning angle,
    target 25deg -- a visibly sharp kink vs. a smooth curve means the
    same thing on any mesh at any scale) with mesh-independent SAFETY
    nets only (2000 iterations, 30% max length decrease -- generous
    enough to never be the thing that decides "done" on a reasonably-
    posed loop, catching only genuine runaway/collapse).

    Returns (adaptive-smooth polyline path, ADAPTIVE_SHORTEN_LENGTH cm)
    -- this REPLACES DELAUNAY_LENGTH as the authoritative circumference:
    the tool's own comment notes DELAUNAY_LENGTH is "likely slightly
    LONGER than the true taut-tape-measure geodesic circumference" since
    it never actually shortens, only retriangulates."""
    out_json = PATHS_DIR / f"{name}.json"
    arg = ",".join(str(v) for v in ring)
    r = subprocess.run(
        [FLIP_GEODESICS, str(OBJ_PATH), f"--measure-loop={arg}", "--units-scale=1",
         "--measure-loop-adaptive-shorten=25",
         f"--export-path={out_json}"],
        capture_output=True, text=True, timeout=30,
    )
    lines = r.stdout.strip().splitlines()
    note_lines = [l for l in r.stderr.strip().splitlines() if l.startswith("NOTE")]
    print(name, "loop", len(ring), "verts ->", lines, note_lines)
    adaptive_len = None
    for line in lines:
        if line.startswith("ADAPTIVE_SHORTEN_LENGTH"):
            adaptive_len = float(line.split()[1]) * M_TO_CM
    adaptive_json = out_json.with_name(out_json.stem + "_adaptive" + out_json.suffix)
    return str(adaptive_json if adaptive_json.exists() else out_json), adaptive_len


BONE = {"LeftArm": 13, "RightArm": 41, "LeftShoulder": 12, "RightShoulder": 40,
        "Hips": 1, "Chest": 4, "LeftLeg": 68, "LeftShin": 69, "RightLeg": 73,
        "Neck1": 5, "Neck2": 6, "LeftForeArm": 14, "LeftHand": 15, "HeadEnd": 8}


def bone_pos(name):
    return rest_bone_poses[0, BONE[name], :3, 3].numpy()


def bone_rot(name):
    return rest_bone_poses[0, BONE[name], :3, :3].numpy()


# --- torso circumference heights ---
hips_z = bone_pos("Hips")[2]
chest_z = bone_pos("Chest")[2]
z_scan = np.linspace(hips_z - 0.05, chest_z + 0.05, 120)
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
between_valid = np.where((z_scan > waist_z) & (z_scan < z_scan[bust_idx]))[0]
between_valid = between_valid[valid[between_valid]]
underbust_idx = between_valid[np.nanargmin(profile[between_valid])] if len(between_valid) > 2 else waist_idx
torso_heights = {"waist": z_scan[waist_idx], "hips": z_scan[hips_idx],
                  "bust": z_scan[bust_idx], "underbust": z_scan[underbust_idx]}

manifest = {}

# circumference loops (real flip_geodesics loops; these are the classic
# tape-measurable circumferences -- a tape physically wrapped around the
# body -- so they always use the real, now-safely-shortened geodesic
# loop (see export_loop()'s docstring for the 2026-11-08 fix: a new
# --measure-loop-bounded-shorten flag in geodesic-probe itself, not a
# plane-cut rendering workaround), never a straight-edge approximation.
loop_lengths = {}
for name, z in torso_heights.items():
    lp = section_loop([0, 0, z], [0, 0, 1], [0, 0, z])
    if lp is not None:
        ring = ring_from_loop(lp)
        manifest[name], loop_lengths[name] = export_loop(name, ring)


def limb_profile(head_pos, tail_pos, n, margin):
    head_pos, tail_pos = np.asarray(head_pos), np.asarray(tail_pos)
    axis = tail_pos - head_pos
    length = np.linalg.norm(axis)
    axis = axis / length
    ts = np.linspace(margin, 1 - margin, n)
    out = []
    for t in ts:
        p = head_pos + t * length * axis
        lp = section_loop(p, axis, p)
        if lp is None:
            out.append((np.nan, None))
            continue
        d = np.linalg.norm(lp.mean(axis=0) - p)
        perim = np.linalg.norm(np.diff(lp, axis=0), axis=1).sum()
        out.append((perim if d < 0.03 else np.nan, lp))
    return out, ts


# leg_circ CORRECTED 2026-11-08: the published method shifts the plane
# DOWN from the crotch until BOTH legs separate into two loops, then
# takes the MINIMUM of the two at that ONE height -- not a scan along
# the limb for a maximum. Tape-measurable circumference -> real
# Delaunay-tightened geodesic loop, not the raw plane-cut perimeter.
hip_pos = bone_pos("LeftLeg")
crotch_z_guess = hip_pos[2] - 0.03
for dz in np.arange(0.0, 0.15, 0.005):
    z = crotch_z_guess - dz
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is not None and len(sec.discrete) >= 2:
        loops_sorted = sorted(sec.discrete, key=lambda lp: -np.linalg.norm(np.diff(lp, axis=0), axis=1).sum())
        if len(loops_sorted) >= 2:
            p0 = np.linalg.norm(np.diff(loops_sorted[0], axis=0), axis=1).sum()
            p1 = np.linalg.norm(np.diff(loops_sorted[1], axis=0), axis=1).sum()
            if p1 > 0.1:
                leg_circ_loop_for_render = loops_sorted[0] if p0 < p1 else loops_sorted[1]
                manifest["leg_circ"], loop_lengths["leg_circ"] = export_loop(
                    "leg_circ", ring_from_loop(leg_circ_loop_for_render))
                break

LEFT_FOREARM_POS = bone_pos("LeftForeArm")
LEFT_HAND_POS = bone_pos("LeftHand")
forearm_profile, forearm_ts = limb_profile(LEFT_FOREARM_POS, LEFT_HAND_POS, 40, 0.05)
forearm_perims = np.array([p for p, _ in forearm_profile])
distal = forearm_ts > 0.66
cand = np.where(distal & ~np.isnan(forearm_perims))[0]
wrist_idx = cand[np.nanargmin(forearm_perims[cand])]
wrist_loop = forearm_profile[wrist_idx][1]
# wrist is tape-measurable (wrapped around the wrist) -- real Delaunay
# loop, not the straight edge-loop-length approximation used 2026-11-08
# before this correction.
manifest["wrist"], loop_lengths["wrist"] = export_loop("wrist", ring_from_loop(wrist_loop))

# open-path fields (real surface geodesics)
v_shoulder = nearest_vertex(bone_pos("LeftArm"))
v_wrist = nearest_vertex(LEFT_HAND_POS)
manifest["arm_length"], arm_length_cm = export_path("arm_length", v_shoulder, v_wrist)

bust_loop = section_loop([0, 0, torso_heights["bust"]], [0, 0, 1], [0, 0, torso_heights["bust"]])
waist_loop = section_loop([0, 0, torso_heights["waist"]], [0, 0, 1], [0, 0, torso_heights["waist"]])
hips_loop = section_loop([0, 0, torso_heights["hips"]], [0, 0, 1], [0, 0, torso_heights["hips"]])

bl_back, br_back = back_side_leftmost_rightmost(bust_loop)
manifest["back_width"], back_width_cm = export_path("back_width", nearest_vertex(bl_back), nearest_vertex(br_back))
wl_back, wr_back = back_side_leftmost_rightmost(waist_loop)
manifest["waist_back_width"], waist_back_width_cm = export_path(
    "waist_back_width", nearest_vertex(wl_back), nearest_vertex(wr_back))
hl_back, hr_back = back_side_leftmost_rightmost(hips_loop)
manifest["hip_back_width"], hip_back_width_cm = export_path(
    "hip_back_width", nearest_vertex(hl_back), nearest_vertex(hr_back))

# shoulder_w, armscye_depth, bust_points, bum_points: REVERTED 2026-11-08
# (iFire: "you must use geodesic for the ones that seem like tape
# masurable") -- a real tailor runs the tape ALONG the body surface for
# every one of these (shoulder-to-shoulder over the trapezius, shoulder
# point down the front to the bust line, nipple-to-nipple across the
# chest, hip-point-to-hip-point across the belly), so they go back to
# the real flip_geodesics surface geodesic, not the published
# reference's literal Euclidean chord -- a deliberate, stated deviation
# from mbotsch/GarmentMeasurements for ones a physical tape would never
# take as a straight line through the air. See CHANGELOG.md.
bl_bust, br_bust = leftmost_rightmost(bust_loop)
bl_hip, br_hip = leftmost_rightmost(hips_loop)
manifest["shoulder_w"], shoulder_w_cm = export_path(
    "shoulder_w", nearest_vertex(bone_pos("LeftArm")), nearest_vertex(bone_pos("RightArm")))
manifest["armscye_depth"], armscye_depth_cm = export_path(
    "armscye_depth", nearest_vertex(bone_pos("LeftShoulder")), nearest_vertex(bl_bust))
manifest["bust_points"], bust_points_cm = export_path(
    "bust_points", nearest_vertex(bl_bust), nearest_vertex(br_bust))
manifest["bum_points"], bum_points_cm = export_path(
    "bum_points", nearest_vertex(bl_hip), nearest_vertex(br_hip))

# neck_w: also tape-measurable (wrapped around the neck) -- real
# Delaunay loop, not the straight edge-loop-length approximation.
neck_z = (bone_pos("Neck1")[2] + bone_pos("Neck2")[2]) / 2
neck_loop = section_loop([0, 0, neck_z], [0, 0, 1], [0, 0, neck_z])
neck_w_cm = None
if neck_loop is not None:
    manifest["neck_w"], neck_w_cm = export_loop("neck_w", ring_from_loop(neck_loop))

with open(OUT_DIR / "paths_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("wrote", OUT_DIR / "paths_manifest.json")

# head_l CORRECTED 2026-11-08: nape-of-neck landmark to crown (the
# reference's measure_head_l), not jaw-to-crown. Not tape-measurable in
# the draped-surface sense (a tailor's height-style measurements are
# taken with a straight rule/stadiometer, not a flexible tape along the
# scalp) -- kept as a real axis extent.
head_top_z = bone_pos("HeadEnd")[2]
head_l_cm = abs(head_top_z - neck_z) * M_TO_CM

vert_bust_line_cm = abs(neck_z - torso_heights["bust"]) * M_TO_CM

# --- angle fields: the EXACT published angle_plane_vec formula ---
neck_pos = (bone_pos("Neck1") + bone_pos("Neck2")) / 2
shoulder_pos = bone_pos("LeftArm")
hips_side_right = rightmost(hips_loop)
waist_side_right = rightmost(waist_loop)

# their Y-up -> ANNY Z-up axis mapping, matching this pipeline's own
# existing convention (Y_theirs=ANNY Z vertical, X_theirs=ANNY X lateral)
shoulder_incl = angle_plane_vec(np.array([0.0, 0.0, -1.0]), shoulder_pos - neck_pos)
hip_inclination = angle_plane_vec(np.array([1.0, 0.0, 0.0]), waist_side_right - hips_side_right)
arm_pose_angle = angle_plane_vec(np.array([0.0, 0.0, -1.0]), LEFT_HAND_POS - shoulder_pos)

# crotch_hip_diff: crotch height (LeftLeg/RightLeg bone midpoint z) vs
# hips ring height, both real -- LandmarkDistanceAxisAligned, unaffected
# by this correction pass.
right_leg_pos = bone_pos("RightLeg")
crotch_z = (bone_pos("LeftLeg")[2] + right_leg_pos[2]) / 2
crotch_hip_diff_cm = abs(crotch_z - torso_heights["hips"]) * M_TO_CM


def _rnd(x):
    return round(x, 2) if x is not None else None


results = {
    "shoulder_w_cm": _rnd(shoulder_w_cm),
    "neck_w_cm": _rnd(neck_w_cm),
    "wrist_cm": _rnd(loop_lengths.get("wrist")),
    "leg_circ_cm": _rnd(loop_lengths.get("leg_circ")),
    "armscye_depth_cm": _rnd(armscye_depth_cm),
    "bust_points_cm": _rnd(bust_points_cm),
    "bum_points_cm": _rnd(bum_points_cm),
    "head_l_cm": _rnd(head_l_cm),
    "vert_bust_line_cm": _rnd(vert_bust_line_cm),
    "shoulder_incl_deg": _rnd(shoulder_incl),
    "hip_inclination_deg": _rnd(hip_inclination),
    "arm_pose_angle_deg": _rnd(arm_pose_angle),
    "crotch_hip_diff_cm": _rnd(crotch_hip_diff_cm),
    "arm_length_cm": _rnd(arm_length_cm),
    "back_width_cm": _rnd(back_width_cm),
    "waist_back_width_cm": _rnd(waist_back_width_cm),
    "hip_back_width_cm": _rnd(hip_back_width_cm),
    "waist_cm": _rnd(loop_lengths.get("waist")),
    "bust_cm": _rnd(loop_lengths.get("bust")),
    "hips_cm": _rnd(loop_lengths.get("hips")),
    "underbust_cm": _rnd(loop_lengths.get("underbust")),
}
print(results)
with open(OUT_DIR / "angle_fields.json", "w") as f:
    json.dump(results, f, indent=2)

# --- layout_hints.json: real positions needed to DRAW the measurement
# diagram (front-view X/Z), so the renderer (Godot, not Python/
# matplotlib -- iFire 2026-11-08: "the svg generator needs to be
# godot") never has to re-derive anatomy on its own. Every number here
# is real data this script already computed above for the real
# measurements themselves (torso_heights, head_top_z, neck_z, bone
# positions) -- this is an export of already-computed real values, not
# new invented positions. The approximate-marker anchors (for the 4
# angle/diff fields and the 2 still-unsettled fields, which have no
# single vertex/curve to anchor to) use the SAME anatomical-height
# offsets the retired render_anny_measurement_wireframe.py used.
layout_hints = {
    "axis_ticks": {
        "head_l": [[0.0, 0.0, head_top_z], [0.0, 0.0, neck_z]],
        "vert_bust_line": [[0.0, 0.0, neck_z], [0.0, 0.0, torso_heights["bust"]]],
    },
    "torso_heights": {k: round(float(v), 5) for k, v in torso_heights.items()},
    "head_top_z": round(float(head_top_z), 5),
    "neck_z": round(float(neck_z), 5),
    # (dx, dz) anatomical marker anchors, dz relative to the named torso
    # height / head_top_z, dx an absolute lateral offset -- the exact
    # heuristic the retired matplotlib script used for its "approx"
    # list, now exported as data instead of hand-tuned Python/GDScript
    # constants duplicated in two places.
    "marker_anchors": {
        "shoulder_incl": {"dx": -0.24, "z_ref": "head_top_z", "dz": -0.19},
        "hip_inclination": {"dx": 0.44, "z_ref": "hips", "dz": -0.02},
        "arm_pose_angle": {"dx": 0.55, "z_ref": "bust", "dz": -0.22},
        "crotch_hip_diff": {"dx": -0.06, "z_ref": "hips", "dz": -0.14},
        "bust_line": {"dx": -0.40, "z_ref": "bust", "dz": 0.03},
        "waist_over_bust_line": {"dx": -0.46, "z_ref": "waist_bust_mid", "dz": 0.0},
    },
}
with open(OUT_DIR / "layout_hints.json", "w") as f:
    json.dump(layout_hints, f, indent=2)
print("wrote", OUT_DIR / "layout_hints.json")
