"""Export REAL curved geodesic polylines (via the actual flip_geodesics CLI,
post-FlipOut-straightening / post-Delaunay) for every settled field that has
a path, for the Godot x-ray contact-sheet render.

Also computes the 4 remaining angle/derived fields (shoulder_incl,
hip_inclination, arm_pose_angle, crotch_hip_diff) from the real ANNY bone
skeleton's own rotation matrices -- not approximated, closing the last gap
from the frontier-learning pass.
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


def back_side_leftmost_rightmost(loop_pts):
    y_mean = loop_pts[:, 1].mean()
    back = loop_pts[loop_pts[:, 1] < y_mean]
    if len(back) < 3:
        back = loop_pts
    i_left = int(np.argmin(back[:, 0]))
    i_right = int(np.argmax(back[:, 0]))
    return back[i_left], back[i_right]


def export_path(name, a, b):
    out_json = PATHS_DIR / f"{name}.json"
    r = subprocess.run(
        [FLIP_GEODESICS, str(OBJ_PATH), f"--measure-path={a},{b}", "--units-scale=1",
         f"--export-path={out_json}"],
        capture_output=True, text=True, timeout=30,
    )
    print(name, "path", a, b, "->", r.stdout.strip().splitlines()[:2])
    return str(out_json)


def export_loop(name, ring):
    out_json = PATHS_DIR / f"{name}.json"
    arg = ",".join(str(v) for v in ring)
    r = subprocess.run(
        [FLIP_GEODESICS, str(OBJ_PATH), f"--measure-loop={arg}", "--units-scale=1",
         f"--export-path={out_json}"],
        capture_output=True, text=True, timeout=30,
    )
    print(name, "loop", len(ring), "verts ->", r.stdout.strip().splitlines()[:2])
    return str(out_json)


BONE = {"LeftArm": 13, "RightArm": 41, "LeftShoulder": 12, "RightShoulder": 40,
        "Hips": 1, "Chest": 4, "LeftLeg": 68, "LeftShin": 69, "Neck1": 5, "Neck2": 6}


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

# circumference loops
for name, z in torso_heights.items():
    lp = section_loop([0, 0, z], [0, 0, 1], [0, 0, z])
    if lp is not None:
        ring = ring_from_loop(lp)
        manifest[name] = export_loop(name, ring)

# leg_circ / wrist
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


hip_pos = bone_pos("LeftLeg")
knee_pos = bone_pos("LeftShin")
thigh_profile, thigh_ts = limb_profile(hip_pos, knee_pos, 40, 0.15)
thigh_perims = [p for p, _ in thigh_profile]
leg_idx = int(np.nanargmax(thigh_perims))
manifest["leg_circ"] = export_loop("leg_circ", ring_from_loop(thigh_profile[leg_idx][1]))

LEFT_FOREARM_POS = rest_bone_poses[0, 14, :3, 3].numpy()
LEFT_HAND_POS = rest_bone_poses[0, 15, :3, 3].numpy()
forearm_profile, forearm_ts = limb_profile(LEFT_FOREARM_POS, LEFT_HAND_POS, 40, 0.05)
forearm_perims = np.array([p for p, _ in forearm_profile])
distal = forearm_ts > 0.66
cand = np.where(distal & ~np.isnan(forearm_perims))[0]
wrist_idx = cand[np.nanargmin(forearm_perims[cand])]
manifest["wrist"] = export_loop("wrist", ring_from_loop(forearm_profile[wrist_idx][1]))

# open-path fields
v_shoulder = nearest_vertex(bone_pos("LeftArm"))
v_wrist = nearest_vertex(LEFT_HAND_POS)
manifest["arm_length"] = export_path("arm_length", v_shoulder, v_wrist)

v_lsh, v_rsh = nearest_vertex(bone_pos("LeftArm")), nearest_vertex(bone_pos("RightArm"))
manifest["shoulder_w"] = export_path("shoulder_w", v_lsh, v_rsh)

neck_z = (bone_pos("Neck1")[2] + bone_pos("Neck2")[2]) / 2
neck_loop = section_loop([0, 0, neck_z], [0, 0, 1], [0, 0, neck_z])
if neck_loop is not None:
    nl, nr = leftmost_rightmost(neck_loop)
    manifest["neck_w"] = export_path("neck_w", nearest_vertex(nl), nearest_vertex(nr))

bust_loop = section_loop([0, 0, torso_heights["bust"]], [0, 0, 1], [0, 0, torso_heights["bust"]])
waist_loop = section_loop([0, 0, torso_heights["waist"]], [0, 0, 1], [0, 0, torso_heights["waist"]])
hips_loop = section_loop([0, 0, torso_heights["hips"]], [0, 0, 1], [0, 0, torso_heights["hips"]])

bl_back, br_back = back_side_leftmost_rightmost(bust_loop)
manifest["back_width"] = export_path("back_width", nearest_vertex(bl_back), nearest_vertex(br_back))
wl_back, wr_back = back_side_leftmost_rightmost(waist_loop)
manifest["waist_back_width"] = export_path("waist_back_width", nearest_vertex(wl_back), nearest_vertex(wr_back))
hl_back, hr_back = back_side_leftmost_rightmost(hips_loop)
manifest["hip_back_width"] = export_path("hip_back_width", nearest_vertex(hl_back), nearest_vertex(hr_back))

bl_bust, br_bust = leftmost_rightmost(bust_loop)
manifest["armscye_depth"] = export_path("armscye_depth", nearest_vertex(bone_pos("LeftShoulder")),
                                         nearest_vertex(bl_bust))
manifest["bust_points"] = export_path("bust_points", nearest_vertex(bl_bust), nearest_vertex(br_bust))
bl_hip, br_hip = leftmost_rightmost(hips_loop)
manifest["bum_points"] = export_path("bum_points", nearest_vertex(bl_hip), nearest_vertex(br_hip))

with open(OUT_DIR / "paths_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("wrote", OUT_DIR / "paths_manifest.json")

# --- real angle/derived fields from the bone skeleton's own rotations ---
def bone_up_axis_angle_from_vertical(rot):
    """Angle (deg) between the bone's local Y axis (rotated into world by
    `rot`) and world-up (Z): 0 = bone points straight up, 90 = horizontal."""
    local_up = np.array([0.0, 1.0, 0.0])
    world_dir = rot @ local_up
    world_dir = world_dir / (np.linalg.norm(world_dir) + 1e-12)
    cos_a = np.clip(np.dot(world_dir, [0, 0, 1]), -1, 1)
    return float(np.degrees(np.arccos(cos_a)))


shoulder_rot = bone_rot("LeftShoulder")
shoulder_incl = bone_up_axis_angle_from_vertical(shoulder_rot)

hips_rot = bone_rot("Hips")
hip_inclination = bone_up_axis_angle_from_vertical(hips_rot)

arm_rot = bone_rot("LeftArm")
# angle between the upper-arm bone direction and the torso's own down
# axis (Chest->Hips) -- "how far the arm is abducted from resting
# alongside the torso", the rest-pose arm_pose_angle.
torso_axis = bone_pos("Hips") - bone_pos("Chest")
torso_axis = torso_axis / np.linalg.norm(torso_axis)
arm_dir = arm_rot @ np.array([0.0, 1.0, 0.0])
arm_dir = arm_dir / (np.linalg.norm(arm_dir) + 1e-12)
cos_a = np.clip(np.dot(arm_dir, torso_axis), -1, 1)
arm_pose_angle = float(np.degrees(np.arccos(cos_a)))

# crotch_hip_diff: crotch height (LeftLeg/RightLeg bone midpoint z) vs
# hips ring height, both real.
right_leg_pos = rest_bone_poses[0, 73, :3, 3].numpy()
crotch_z = (bone_pos("LeftLeg")[2] + right_leg_pos[2]) / 2
crotch_hip_diff_cm = abs(crotch_z - torso_heights["hips"]) * M_TO_CM

angle_results = {
    "shoulder_incl_deg": round(shoulder_incl, 2),
    "hip_inclination_deg": round(hip_inclination, 2),
    "arm_pose_angle_deg": round(arm_pose_angle, 2),
    "crotch_hip_diff_cm": round(crotch_hip_diff_cm, 2),
}
print(angle_results)
with open(OUT_DIR / "angle_fields.json", "w") as f:
    json.dump(angle_results, f, indent=2)
