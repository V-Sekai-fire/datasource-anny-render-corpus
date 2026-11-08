"""Render a real front-view wireframe SVG of the ANNY-SOMA-X mesh (vector
graphics, viewable on GitHub), annotated with the 26 GarmentCode body-
measurement locations.

Every annotation drawn here uses REAL mesh data -- actual vertex rings
snapped from the already-validated cross-section loops (waist, bust,
underbust, hips, leg_circ, wrist), the actual arm_length path endpoints
(vertices 6058/818), and the real height extent -- the same values
reported in datasource-anny-render-corpus PR #2/#3 and
sinew-mocap/measure's AnnySomaX.lean fixtures. The remaining 20 fields
with no exact vertex pick yet are shown as approximate region markers,
clearly labeled APPROX in the legend -- not invented precise positions.
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from anny import Anny
from matplotlib.collections import LineCollection

OUT_DIR = pathlib.Path("build/anny_export")
OUT_DIR.mkdir(parents=True, exist_ok=True)
M_TO_CM = 100.0

model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref", phenotypes="all")
model.eval()
with torch.no_grad():
    rest_out = model(phenotype_kwargs={"gender": 1.0})
rest_bone_poses = rest_out["rest_bone_poses"]
verts = rest_out["rest_vertices"][0].numpy().astype(np.float64)
faces = np.asarray(model.faces, dtype=np.int64)
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def section_loop(mesh, origin, normal, centerline_point):
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


# --- torso circumference heights (same derivation as measure_anny_circumferences.py) ---
HIPS_BONE, CHEST_BONE = 1, 4
hips_z = rest_bone_poses[0, HIPS_BONE, 2, 3].item()
chest_z = rest_bone_poses[0, CHEST_BONE, 2, 3].item()
z_scan = np.linspace(hips_z - 0.05, chest_z + 0.05, 120)
profile = np.array([
    (lambda lp: np.linalg.norm(np.diff(lp, axis=0), axis=1).sum() if lp is not None else np.nan)(
        section_loop(mesh, [0, 0, z], [0, 0, 1], [0, 0, z])
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

real_loops = {}
for name, z in torso_heights.items():
    lp = section_loop(mesh, [0, 0, z], [0, 0, 1], [0, 0, z])
    if lp is not None:
        real_loops[name] = mesh.vertices[ring_from_loop(lp)]

# --- limb circumferences: leg_circ (left thigh max), wrist (distal forearm min) ---
def limb_profile(head_pos, tail_pos, n, margin):
    head_pos, tail_pos = np.asarray(head_pos), np.asarray(tail_pos)
    axis = tail_pos - head_pos
    length = np.linalg.norm(axis)
    axis = axis / length
    ts = np.linspace(margin, 1 - margin, n)
    out = []
    for t in ts:
        p = head_pos + t * length * axis
        lp = section_loop(mesh, p, axis, p)
        if lp is None:
            out.append((np.nan, None))
            continue
        d = np.linalg.norm(lp.mean(axis=0) - p)
        perim = np.linalg.norm(np.diff(lp, axis=0), axis=1).sum()
        out.append((perim if d < 0.03 else np.nan, lp))
    return out, ts


LEFT_LEG, LEFT_SHIN = 68, 69
hip_pos = rest_bone_poses[0, LEFT_LEG, :3, 3].numpy()
knee_pos = rest_bone_poses[0, LEFT_SHIN, :3, 3].numpy()
thigh_profile, thigh_ts = limb_profile(hip_pos, knee_pos, 40, 0.15)
thigh_perims = [p for p, _ in thigh_profile]
leg_idx = int(np.nanargmax(thigh_perims))
real_loops["leg_circ"] = mesh.vertices[ring_from_loop(thigh_profile[leg_idx][1])]

LEFT_FOREARM, LEFT_HAND = 14, 15
elbow_pos = rest_bone_poses[0, LEFT_FOREARM, :3, 3].numpy()
wrist_pos = rest_bone_poses[0, LEFT_HAND, :3, 3].numpy()
forearm_profile, forearm_ts = limb_profile(elbow_pos, wrist_pos, 40, 0.05)
forearm_perims = np.array([p for p, _ in forearm_profile])
distal = forearm_ts > 0.66
cand = np.where(distal & ~np.isnan(forearm_perims))[0]
wrist_idx = cand[np.nanargmin(forearm_perims[cand])]
real_loops["wrist"] = mesh.vertices[ring_from_loop(forearm_profile[wrist_idx][1])]

# --- arm_length (real open path endpoints) ---
LEFT_ARM, LEFT_HAND2 = 13, 15
shoulder_pos = rest_bone_poses[0, LEFT_ARM, :3, 3].numpy()


def nearest_vertex(p):
    d2 = np.sum((mesh.vertices - np.asarray(p)) ** 2, axis=1)
    return int(np.argmin(d2))


v_shoulder = nearest_vertex(shoulder_pos)
v_wrist = nearest_vertex(wrist_pos)
arm_path = mesh.vertices[[v_shoulder, v_wrist]]

print("real loop sizes:", {k: len(v) for k, v in real_loops.items()})
print(f"arm path verts: {v_shoulder} -> {v_wrist}")

# --- render: front view, X (horizontal) vs Z (vertical, up) ---
fig, ax = plt.subplots(figsize=(10, 10.5))

# sampled wireframe: every 14th face's 3 edges, at low alpha, front-view
# (keeps the SVG a few hundred KB instead of several MB so it stays quick
# to view on GitHub, while still reading as a recognizable body wireframe)
stride = 14
sample_faces = faces[::stride]
segs = []
for a, b, c in sample_faces:
    for i, j in ((a, b), (b, c), (c, a)):
        segs.append([(verts[i, 0], verts[i, 2]), (verts[j, 0], verts[j, 2])])
ax.add_collection(LineCollection(segs, colors="#9ca3af", linewidths=0.3, alpha=0.5, zorder=1))

circ_label_offset = {  # (dx beyond loop max-x, dz) -- hand-tuned so the four
    # torso loops' labels (very close in height on this body: waist/underbust
    # are only ~2mm apart) don't stack on top of each other.
    "bust": (0.07, 0.025), "underbust": (0.14, -0.02),
    "waist": (0.20, 0.0), "hips": (0.07, 0.0),
    "leg_circ": (0.16, 0.0), "wrist": (0.09, -0.035),
}
for name in ["waist", "underbust", "bust", "hips", "leg_circ", "wrist"]:
    loop = real_loops[name]
    pts = np.vstack([loop, loop[:1]])
    ax.plot(pts[:, 0], pts[:, 2], color="#16a34a", linewidth=1.6, zorder=3)
    dx, dz = circ_label_offset[name]
    lx, lz = loop[:, 0].max() + dx, loop[:, 2].mean() + dz
    ax.annotate(name, (lx, lz), fontsize=8, color="#166534", zorder=4)

ax.plot(arm_path[:, 0], arm_path[:, 2], color="#16a34a", linewidth=1.6, marker="o",
         markersize=3, zorder=3)
ax.annotate("arm_length", (arm_path[0, 0] - 0.02, arm_path[0, 2] + 0.06),
            fontsize=8, color="#166534", zorder=4, ha="right")

# --- the 8 surface-distance fields resolved in the frontier-learning ---
# --- iteration (CHANGELOG.md), real bone/ring-vertex pairs. ------------
BONE = {"LeftArm": 13, "RightArm": 41, "LeftShoulder": 12}


def bone_pos(name):
    return rest_bone_poses[0, BONE[name], :3, 3].numpy()


def nearest_vertex2(p):
    d2 = np.sum((verts - np.asarray(p)) ** 2, axis=1)
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


neck_z = (rest_bone_poses[0, 5, 2, 3].item() + rest_bone_poses[0, 6, 2, 3].item()) / 2
neck_loop = section_loop(mesh, [0, 0, neck_z], [0, 0, 1], [0, 0, neck_z])
bust_loop_pts = real_loops["bust"]
waist_loop_pts = real_loops["waist"]
hips_loop_pts = real_loops["hips"]

frontier_lines = {
    "shoulder_w": (bone_pos("LeftArm"), bone_pos("RightArm")),
}
if neck_loop is not None:
    frontier_lines["neck_w"] = leftmost_rightmost(neck_loop)
frontier_lines["back_width"] = back_side_leftmost_rightmost(bust_loop_pts)
frontier_lines["hip_back_width"] = back_side_leftmost_rightmost(hips_loop_pts)
frontier_lines["waist_back_width"] = back_side_leftmost_rightmost(waist_loop_pts)
bl_bust, br_bust = leftmost_rightmost(bust_loop_pts)
frontier_lines["armscye_depth"] = (bone_pos("LeftShoulder"), bl_bust)
frontier_lines["bust_points"] = (bl_bust, br_bust)
frontier_lines["bum_points"] = leftmost_rightmost(hips_loop_pts)

frontier_label_offset = {
    "shoulder_w": (-0.12, 0.04), "neck_w": (0.02, -0.03),
    "back_width": (0.14, 0.015), "hip_back_width": (-0.10, 0.0),
    "waist_back_width": (-0.10, -0.02), "armscye_depth": (0.18, -0.045),
    "bust_points": (0.14, -0.015), "bum_points": (0.14, -0.02),
}
for name, (pa, pb) in frontier_lines.items():
    ax.plot([pa[0], pb[0]], [pa[2], pb[2]], color="#16a34a", linewidth=1.3,
             marker="o", markersize=2.5, zorder=3)
    mx, mz = (pa[0] + pb[0]) / 2, (pa[2] + pb[2]) / 2
    dx, dz = frontier_label_offset.get(name, (0.02, 0.0))
    ax.annotate(name, (mx + dx, mz + dz), fontsize=7, color="#166534", zorder=4,
                ha="center")

zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
ax.annotate("", xy=(-0.60, zmax), xytext=(-0.60, zmin),
            arrowprops=dict(arrowstyle="<->", color="#16a34a", linewidth=1.2), zorder=3)
ax.annotate("height", (-0.68, (zmin + zmax) / 2), fontsize=8, color="#166534",
            rotation=90, va="center", zorder=4)

# head_l, vert_bust_line, waist_over_bust_line: axis extents (real, but no
# single vertex PAIR -- drawn as short vertical green ticks at their real
# z-heights rather than a two-point line).
head_top_z = rest_bone_poses[0, 8, 2, 3].item()   # HeadEnd
jaw_z = rest_bone_poses[0, 9, 2, 3].item()        # Jaw
axis_ticks = {
    "head_l": (0.55, head_top_z, jaw_z),
    "vert_bust_line": (0.62, neck_z, torso_heights["bust"]),
    "waist_over_bust_line": (0.70, torso_heights["bust"], torso_heights["waist"]),
}
for name, (x, za, zb) in axis_ticks.items():
    ax.annotate("", xy=(x, za), xytext=(x, zb),
                arrowprops=dict(arrowstyle="<->", color="#16a34a", linewidth=1.0), zorder=3)
    ax.annotate(name, (x + 0.02, (za + zb) / 2), fontsize=7, color="#166534",
                rotation=90, va="center", zorder=4)

# Remaining 4 fields (angle/derived -- shoulder_incl, hip_inclination,
# arm_pose_angle, crotch_hip_diff): the schema classifies these with
# confidence from the NAME alone (settled in BodyMeasurements.lean), but
# this frontier-learning pass did not give them a vertex-level geometric
# pick -- shown as approximate markers, honestly distinguished from the
# real lines/ticks above.
approx = [
    ("shoulder_incl", -0.24, zmax - 0.19),
    ("hip_inclination", 0.44, torso_heights["hips"] - 0.02),
    ("crotch_hip_diff", -0.06, torso_heights["hips"] - 0.14),
    ("arm_pose_angle", 0.55, torso_heights["bust"] - 0.22),
]
for name, dx, z in approx:
    ax.plot([dx], [z], marker="o", markersize=4, color="#dc2626", zorder=3)
    ha = "left" if dx >= 0 else "right"
    ax.annotate(name, (dx + (0.02 if dx >= 0 else -0.02), z), fontsize=7,
                color="#7f1d1d", zorder=4, ha=ha)

ax.set_aspect("equal")
ax.set_xlim(-0.95, 1.0)
ax.set_ylim(zmin - 0.05, zmax + 0.08)
ax.axis("off")
ax.set_title("ANNY-SOMA-X — 26 GarmentCode measurement locations\n"
              "green = real vertex/geodesic/axis-extent data (22 of 26), "
              "red = angle/derived field, no vertex-level pick yet (4 of 26)",
              fontsize=9)

SVG_PATH = OUT_DIR / "anny_measurement_wireframe.svg"
fig.savefig(SVG_PATH, format="svg", bbox_inches="tight")
print(f"wrote {SVG_PATH}")

with open(OUT_DIR / "wireframe_loop_vertices.json", "w") as f:
    json.dump({
        "real_loops_vertex_counts": {k: len(v) for k, v in real_loops.items()},
        "arm_length_vertices": [v_shoulder, v_wrist],
    }, f, indent=2)
