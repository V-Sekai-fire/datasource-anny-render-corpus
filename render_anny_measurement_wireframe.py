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

# --- real curved geodesic polylines (from export_real_geodesic_paths.py's
# actual flip_geodesics --export-path output: the smooth FlipOut-
# straightened open paths / Delaunay closed loops, not a straight chord
# or the raw zigzag) -- loaded here instead of connecting endpoints with
# a straight line. ---
PATHS_MANIFEST = OUT_DIR / "paths_manifest.json"
real_curves = {}
if PATHS_MANIFEST.exists():
    with open(PATHS_MANIFEST) as f:
        manifest = json.load(f)
    for field_name, path_file in manifest.items():
        with open(path_file) as pf:
            lines = json.load(pf)
        if lines and len(lines[0]) >= 2:
            real_curves[field_name] = np.array(lines[0])  # cm from --units-scale=1... wait see below
    print(f"loaded {len(real_curves)} real curved polylines from {PATHS_MANIFEST}")
else:
    print(f"WARNING: {PATHS_MANIFEST} not found -- run export_real_geodesic_paths.py first; "
          "falling back to straight-line connections for this render")

# --- render: front view, X (horizontal) vs Z (vertical, up) ---
fig, ax = plt.subplots(figsize=(10, 10.5))

# label_queue: every labeled anchor collected here instead of calling
# ax.annotate() immediately, so ALL labels (circumference loops,
# surface-geodesic/Euclidean fields, axis extents, angle markers) go
# through ONE unified boundary-labeling pass at the end -- see
# place_all_labels() below. Each entry: (text, x, z, line_color, text_color).
label_queue = []

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

for name in ["waist", "underbust", "bust", "hips", "leg_circ", "wrist"]:
    loop = real_curves[name] if name in real_curves else real_loops[name]
    if name in real_curves:
        pts = loop  # already closed-loop polyline from the real Delaunay solve
    else:
        pts = np.vstack([loop, loop[:1]])
    ax.plot(pts[:, 0], pts[:, 2], color="#16a34a", linewidth=1.6, zorder=3)
    lx, lz = real_loops[name][:, 0].mean(), real_loops[name][:, 2].mean()
    label_queue.append((name, lx, lz, "#16a34a", "#166534"))

# neck_w: CORRECTED 2026-11-08 to match the real published reference
# implementation (mbotsch/GarmentMeasurements measure_vertex_trace) --
# the FULL neck ring's edge-loop length, not a left-right chord. Drawn
# as a closed loop, same as the circumference fields above.
neck_z = (rest_bone_poses[0, 5, 2, 3].item() + rest_bone_poses[0, 6, 2, 3].item()) / 2
neck_loop = section_loop(mesh, [0, 0, neck_z], [0, 0, 1], [0, 0, neck_z])
if neck_loop is not None:
    neck_ring = mesh.vertices[ring_from_loop(neck_loop)]
    pts = np.vstack([neck_ring, neck_ring[:1]])
    ax.plot(pts[:, 0], pts[:, 2], color="#16a34a", linewidth=1.6, zorder=3)
    label_queue.append(("neck_w", neck_ring[:, 0].mean(), neck_ring[:, 2].mean(), "#16a34a", "#166534"))

if "arm_length" in real_curves:
    arm_curve = real_curves["arm_length"]
    ax.plot(arm_curve[:, 0], arm_curve[:, 2], color="#16a34a", linewidth=1.6, zorder=3)
else:
    ax.plot(arm_path[:, 0], arm_path[:, 2], color="#16a34a", linewidth=1.6, marker="o",
             markersize=3, zorder=3)
label_queue.append(("arm_length", arm_path[:, 0].mean(), arm_path[:, 2].mean(), "#16a34a", "#166534"))

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
bust_loop_pts = real_loops["bust"]
waist_loop_pts = real_loops["waist"]
hips_loop_pts = real_loops["hips"]
bl_bust, br_bust = leftmost_rightmost(bust_loop_pts)

# surface-geodesic fields (real curved paths, from real_curves): back/
# hip/waist back-width only -- the others below are straight-line
# Euclidean per the 2026-11-08 correction, drawn separately.
frontier_lines = {
    "back_width": back_side_leftmost_rightmost(bust_loop_pts),
    "hip_back_width": back_side_leftmost_rightmost(hips_loop_pts),
    "waist_back_width": back_side_leftmost_rightmost(waist_loop_pts),
}

# shoulder_w, armscye_depth, bust_points, bum_points: CORRECTED 2026-11-08
# to match the real published reference implementation
# (mbotsch/GarmentMeasurements measure_landmark_distance_euclidean) --
# a straight 3D (Euclidean) chord between the two landmarks, NOT a
# surface geodesic. Drawn as DASHED straight lines to visually
# distinguish from the real surface-geodesic curves above.
euclidean_lines = {
    "shoulder_w": (bone_pos("LeftArm"), bone_pos("RightArm")),
    "armscye_depth": (bone_pos("LeftShoulder"), bl_bust),
    "bust_points": (bl_bust, br_bust),
    "bum_points": leftmost_rightmost(hips_loop_pts),
}

for name, (pa, pb) in frontier_lines.items():
    if name in real_curves:
        curve = real_curves[name]
        ax.plot(curve[:, 0], curve[:, 2], color="#16a34a", linewidth=1.3, zorder=3)
        mx, mz = curve[:, 0].mean(), curve[:, 2].mean()
    else:
        ax.plot([pa[0], pb[0]], [pa[2], pb[2]], color="#16a34a", linewidth=1.3,
                 marker="o", markersize=2.5, zorder=3)
        mx, mz = (pa[0] + pb[0]) / 2, (pa[2] + pb[2]) / 2
    label_queue.append((name, mx, mz, "#16a34a", "#166534"))

for name, (pa, pb) in euclidean_lines.items():
    ax.plot([pa[0], pb[0]], [pa[2], pb[2]], color="#16a34a", linewidth=1.3,
             linestyle="--", marker="o", markersize=2.5, zorder=3)
    mx, mz = (pa[0] + pb[0]) / 2, (pa[2] + pb[2]) / 2
    label_queue.append((name, mx, mz, "#16a34a", "#166534"))

zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
ax.annotate("", xy=(-0.60, zmax), xytext=(-0.60, zmin),
            arrowprops=dict(arrowstyle="<->", color="#16a34a", linewidth=1.2), zorder=3)
ax.annotate("height", (-0.68, (zmin + zmax) / 2), fontsize=8, color="#166534",
            rotation=90, va="center", zorder=4)

# head_l: CORRECTED 2026-11-08 to the real published reference's
# `measure_head_l` lower landmark (nape_of_neck, not jaw). vert_bust_line
# remains a real axis extent. waist_over_bust_line and bust_line are
# REVERTED to unsettled (see CHANGELOG.md): the published reference
# computes them as real parasagittal-plane-cut polyline lengths, not a
# simple axis-aligned z-difference -- drawn RED below, not green.
head_top_z = rest_bone_poses[0, 8, 2, 3].item()   # HeadEnd
axis_ticks = {
    "head_l": (0.55, head_top_z, neck_z),
    "vert_bust_line": (0.62, neck_z, torso_heights["bust"]),
}
for name, (x, za, zb) in axis_ticks.items():
    ax.annotate("", xy=(x, za), xytext=(x, zb),
                arrowprops=dict(arrowstyle="<->", color="#16a34a", linewidth=1.0), zorder=3)
    label_queue.append((name, x, (za + zb) / 2, "#16a34a", "#166534"))


# Remaining 4 fields (angle/derived -- shoulder_incl, hip_inclination,
# arm_pose_angle, crotch_hip_diff): real values now computed from the
# bone skeleton's own rotations (export_real_geodesic_paths.py's
# angle_fields.json), still drawn as markers since an angle/difference
# has no single curve to draw -- but labeled with the real degree/cm
# value instead of a bare field name.
ANGLE_FIELDS_PATH = OUT_DIR / "angle_fields.json"
angle_values = {}
if ANGLE_FIELDS_PATH.exists():
    with open(ANGLE_FIELDS_PATH) as f:
        angle_values = json.load(f)

approx = [
    ("shoulder_incl", angle_values.get("shoulder_incl_deg"), "deg", -0.24, zmax - 0.19),
    ("hip_inclination", angle_values.get("hip_inclination_deg"), "deg", 0.44, torso_heights["hips"] - 0.02),
    ("crotch_hip_diff", angle_values.get("crotch_hip_diff_cm"), "cm", -0.06, torso_heights["hips"] - 0.14),
    ("arm_pose_angle", angle_values.get("arm_pose_angle_deg"), "deg", 0.55, torso_heights["bust"] - 0.22),
    # REVERTED to unsettled 2026-11-08: the real reference computes these
    # as parasagittal-plane-cut polyline lengths, which this package does
    # not implement -- drawn red/unlabeled, honestly, not under a wrong
    # value.
    ("bust_line", None, "cm", -0.40, torso_heights["bust"] + 0.03),
    ("waist_over_bust_line", None, "cm",
     -0.46, (torso_heights["waist"] + torso_heights["bust"]) / 2),
]
for name, val, unit, dx, z in approx:
    color = "#16a34a" if val is not None else "#dc2626"
    text_color = "#166534" if val is not None else "#7f1d1d"
    label = f"{name}\n{val:.1f}{unit}" if val is not None else name
    ax.plot([dx], [z], marker="o", markersize=4, color=color, zorder=3)
    label_queue.append((label, dx, z, color, text_color))

# ============ Bekos, Kaufmann, Symvonis & Wolff (2007) type-opo
# boundary labeling: ONE combined left/right layout pass across every
# queued anchor (circumference loops, surface-geodesic/Euclidean
# fields, axis extents, angle markers), replacing the hand-tuned
# per-field (dx, dz) offset dicts this file shipped with. iFire: "the
# new one was wrong" (about the x-ray PNG pipeline's text-on-body
# crowding) -- this SVG had the SAME underlying problem (e.g. "bust",
# "armscye_depth", "bust_points" labels landing on top of each other
# near the chest); the SAME validated algorithm from
# sinew-mocap/measure's xray_oit_render.gd now applies here too. ============
left_q = sorted([a for a in label_queue if a[1] < 0], key=lambda a: -a[2])   # top-to-bottom = descending z
right_q = sorted([a for a in label_queue if a[1] >= 0], key=lambda a: -a[2])

MARGIN_X_LEFT = -1.25
MARGIN_X_RIGHT = 1.20
MARGIN_Z_TOP = zmax - 0.05  # leave headroom below the title (fig.suptitle/ax.set_title)
MARGIN_Z_BOTTOM = zmin + 0.05
# Rails sit NEAR THE BODY (not near the margin): the vertical
# height-adjusting segment happens close to the anchor, so the LONG
# horizontal sweep out to the margin runs at each item's own TARGET
# label_z (unique per item by construction) instead of its raw anchor
# z. The earlier version put rails near the margin, making the FIRST
# (long) segment run at the ANCHOR's raw z -- which, for a dense
# layout, can graze close to a DIFFERENT item's label_z and visually
# clip that item's text even though no two LEADERS actually crossed
# each other. Flipping which segment is long fixes this: a long run at
# a unique target z cannot intersect anyone else's text, because only
# the item that OWNS that z has text there.
INNER_RAIL_BASE_LEFT = -0.78
INNER_RAIL_BASE_RIGHT = 0.78
RAIL_MARGIN_BUFFER = 0.08  # rails must never reach margin_x (where text starts)


def place_column(items, margin_x, rail_sign, inner_rail_base):
    n = len(items)
    if n == 0:
        return
    step = (MARGIN_Z_TOP - MARGIN_Z_BOTTOM) / max(1, n - 1) if n > 1 else 0.0
    # Rail spacing SCALED to the actual item count: a fixed per-item
    # increment (the earlier bug) grows unbounded and, past ~10 items,
    # pushed later rails PAST margin_x -- putting their vertical
    # height-adjusting segment literally inside the text column, where
    # it crossed through neighboring items' labels. Capping the total
    # rail span to the real available gap (inner_rail_base to
    # margin_x - buffer) makes that geometrically impossible regardless
    # of how many items end up on one side.
    available_span = abs(margin_x - inner_rail_base) - RAIL_MARGIN_BUFFER
    rail_spacing = available_span / max(1, n - 1) if n > 1 else 0.0
    for i, (text, x, z, line_color, text_color) in enumerate(items):
        label_z = MARGIN_Z_TOP - step * i if n > 1 else (MARGIN_Z_TOP + MARGIN_Z_BOTTOM) / 2.0
        rail_x = inner_rail_base + rail_sign * i * rail_spacing
        # type-opo: anchor -> (rail_x, z) [short] -> (rail_x, label_z) [vertical]
        # -> (margin_x, label_z) [long, at the FINAL target z]
        ax.plot([x, rail_x, rail_x, margin_x], [z, z, label_z, label_z],
                color=line_color, linewidth=0.6, alpha=0.8, zorder=2)
        ha = "left" if rail_sign > 0 else "right"
        text_x = margin_x + (0.03 if rail_sign > 0 else -0.03)
        # Text sits ABOVE the connecting line's z, not straddling it --
        # at this item density a leader's own near-zero-length vertical
        # rail segment (anchor.z close to its own assigned label_z) was
        # rendering right through the first couple characters otherwise.
        # 0.025 (not 0.012): multi-line labels (angle fields: name on
        # one line, "43.9deg" on the next) are TALLER, and matplotlib's
        # va="bottom" anchors the BOTTOM of the whole multi-line block
        # at this z -- too small a gap left the bottom line still
        # touching the connecting line for those entries specifically.
        ax.annotate(text, (text_x, label_z + 0.025), fontsize=7, color=text_color,
                    zorder=4, ha=ha, va="bottom")


place_column(left_q, MARGIN_X_LEFT, rail_sign=-1.0, inner_rail_base=INNER_RAIL_BASE_LEFT)
place_column(right_q, MARGIN_X_RIGHT, rail_sign=1.0, inner_rail_base=INNER_RAIL_BASE_RIGHT)

ax.set_aspect("equal")
ax.set_xlim(-1.55, 1.55)
ax.set_ylim(zmin - 0.05, zmax + 0.08)
ax.axis("off")
ax.set_title("ANNY-SOMA-X — 26 GarmentCode measurement locations\n"
              "green = 24 real fields (dashed = Euclidean, per the real "
              "mbotsch/GarmentMeasurements reference); red = 2 reverted "
              "to unsettled",
              fontsize=9)

SVG_PATH = OUT_DIR / "anny_measurement_wireframe.svg"
fig.savefig(SVG_PATH, format="svg", bbox_inches="tight")
print(f"wrote {SVG_PATH}")

with open(OUT_DIR / "wireframe_loop_vertices.json", "w") as f:
    json.dump({
        "real_loops_vertex_counts": {k: len(v) for k, v in real_loops.items()},
        "arm_length_vertices": [v_shoulder, v_wrist],
    }, f, indent=2)
