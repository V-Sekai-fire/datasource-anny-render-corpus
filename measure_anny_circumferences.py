"""Compute the 6 GarmentCode circumference measurements from the ANNY-SOMA-X
rest mesh via mesh-plane cross-sections -- the mechanically well-defined
subset of the 26-field body_params.py schema (torso circumferences = global-Z
slices since the torso's long axis is Z; limb circumferences = slices
perpendicular to the limb's own bone axis, from rest_bone_poses).

Definitions are from GarmentCode's own docs/Body Measurements GarmentCode.pdf:
  waist:      circumference, ~middle between ribs and hip bones
  bust:       circumference, parallel to floor, through the bust points
  underbust:  circumference, parallel to floor, right below the bust
  hips:       circumference, parallel to floor, through the bum points
  leg_circ:   circumference of the thickest part of a thigh, one leg
  wrist:      circumference, thinnest part of the arm at the root of the hand

NOT computed here (need geodesic/landmark measurements beyond a plain plane
cross-section -- left for a follow-up): waist_back_width, back_width,
hip_back_width, neck_w, waist_line, waist_over_bust_line, bust_line,
arm_length, height (trivial, already have it), head_l, shoulder_w,
armscye_depth, bust_points, hips_line, bum_points, crotch_hip_diff,
vert_bust_line, hip_inclination, shoulder_incl, arm_pose_angle.
"""
import numpy as np
import torch
import trimesh
from anny import Anny

model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref", phenotypes="all")
model.eval()
with torch.no_grad():
    rest_out = model(phenotype_kwargs={"gender": 1.0})
rest_bone_poses = rest_out["rest_bone_poses"]
verts = rest_out["rest_vertices"][0].numpy().astype(np.float64)
faces = np.asarray(model.faces, dtype=np.int64)
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

M_TO_CM = 100.0


def section_perimeter(mesh, origin, normal, centerline_point=None):
    """Perimeter of the ONE section loop nearest centerline_point (default:
    the plane origin projected to x=y=0) -- the torso/limb loop, discarding
    other loops the same plane cuts through (e.g. the arms hanging past the
    torso at chest-to-hip height in this rest pose; confirmed by inspection:
    those heights return 3 loops, not 1 -- a flat 'must be exactly 1 loop'
    check was wrong and silently failed every slice before this fix).
    Also returns the chosen loop's centroid distance to centerline_point, so
    callers can reject a slice where even the NEAREST loop is actually far
    away (e.g. a limb scan point still inside the torso/pelvis, where the
    only loop available is the torso's, not the limb's)."""
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None or len(sec.discrete) == 0:
        return None, 0, np.inf
    loops = sec.discrete
    if centerline_point is None:
        centerline_point = np.array([0.0, 0.0, origin[2]]) if len(origin) == 3 else np.zeros(3)
    centroids = [loop.mean(axis=0) for loop in loops]
    dists = [np.linalg.norm(c - centerline_point) for c in centroids]
    best = int(np.argmin(dists))
    chosen = loops[best]
    d = np.diff(chosen, axis=0)
    total = np.linalg.norm(d, axis=1).sum()
    if not np.allclose(chosen[0], chosen[-1]):
        total += np.linalg.norm(chosen[0] - chosen[-1])
    return total, len(loops), dists[best]


# --- Torso circumferences: slice along global Z (confirmed torso axis) ---
HIPS_BONE, CHEST_BONE = 1, 4  # Hips, Chest
hips_z = rest_bone_poses[0, HIPS_BONE, 2, 3].item()
chest_z = rest_bone_poses[0, CHEST_BONE, 2, 3].item()
print(f"torso scan band: hips_z={hips_z:.4f}  chest_z={chest_z:.4f}")

z_scan = np.linspace(hips_z - 0.05, chest_z + 0.05, 120)
profile = []
for z in z_scan:
    perim, n_loops, dist = section_perimeter(mesh, origin=[0, 0, z], normal=[0, 0, 1])
    # Not "exactly 1 loop" (see section_perimeter's docstring -- the arms
    # add 2 more loops through most of this band); just require we got a
    # loop at all.
    profile.append(perim if perim is not None else np.nan)
profile = np.array(profile)

valid = ~np.isnan(profile)
if valid.sum() < 10:
    raise SystemExit(f"CONTROL FAILED: too few clean torso slices ({valid.sum()}/120)")

# Waist: global minimum perimeter in the clean torso band.
waist_idx = np.nanargmin(profile)
waist_z = z_scan[waist_idx]
waist_circ = profile[waist_idx] * M_TO_CM

# Hips: local maximum BELOW the waist (bum region).
below = np.where(z_scan < waist_z)[0]
below_valid = below[valid[below]]
if len(below_valid) == 0:
    raise SystemExit("CONTROL FAILED: no valid torso slice below the waist for hips")
hips_idx = below_valid[np.nanargmax(profile[below_valid])]
hips_circ = profile[hips_idx] * M_TO_CM

# Bust: local maximum ABOVE the waist (chest region).
above = np.where(z_scan > waist_z)[0]
above_valid = above[valid[above]]
if len(above_valid) == 0:
    raise SystemExit("CONTROL FAILED: no valid torso slice above the waist for bust")
bust_idx = above_valid[np.nanargmax(profile[above_valid])]
bust_circ = profile[bust_idx] * M_TO_CM

# Underbust: local minimum between waist and bust (a notch, may be shallow
# or absent on a low-detail mesh -- report it but flag if it isn't actually
# a local min, i.e. the band is monotonic).
between = np.where((z_scan > waist_z) & (z_scan < z_scan[bust_idx]))[0]
between_valid = between[valid[between]]
if len(between_valid) > 2:
    underbust_idx = between_valid[np.nanargmin(profile[between_valid])]
    underbust_circ = profile[underbust_idx] * M_TO_CM
    underbust_is_interior_min = (underbust_idx != between_valid[0] and underbust_idx != between_valid[-1])
else:
    underbust_idx = between_valid[0] if len(between_valid) else waist_idx
    underbust_circ = profile[underbust_idx] * M_TO_CM
    underbust_is_interior_min = False

print(f"waist:     z={z_scan[waist_idx]:.4f}  circ={waist_circ:.2f} cm")
print(f"hips:      z={z_scan[hips_idx]:.4f}  circ={hips_circ:.2f} cm")
print(f"bust:      z={z_scan[bust_idx]:.4f}  circ={bust_circ:.2f} cm")
print(f"underbust: z={z_scan[underbust_idx]:.4f}  circ={underbust_circ:.2f} cm  "
      f"(interior local min: {underbust_is_interior_min})")

# PLANTED CONTROLS: basic tailoring sanity that must hold for any adult
# female body, regardless of phenotype -- catches a slicing-axis or
# loop-selection bug rather than trusting whatever numbers come out.
if not (hips_circ > waist_circ):
    raise SystemExit(f"CONTROL FAILED: hips ({hips_circ:.2f}) not > waist ({waist_circ:.2f})")
if not (bust_circ > waist_circ):
    raise SystemExit(f"CONTROL FAILED: bust ({bust_circ:.2f}) not > waist ({waist_circ:.2f})")
if not (60 < waist_circ < 120):
    raise SystemExit(f"CONTROL FAILED: waist {waist_circ:.2f} cm outside plausible adult range")
if not (70 < hips_circ < 140):
    raise SystemExit(f"CONTROL FAILED: hips {hips_circ:.2f} cm outside plausible adult range")
if not (70 < bust_circ < 140):
    raise SystemExit(f"CONTROL FAILED: bust {bust_circ:.2f} cm outside plausible adult range")
print("CONTROL PASSED: hips > waist, bust > waist, all three in plausible adult range")


# --- Limb circumferences: slice perpendicular to the limb's own bone axis ---
def limb_circumference_profile(mesh, head_pos, tail_pos, n_slices, margin=0.15):
    """Scan perpendicular cross-sections along a limb segment, margin-trimmed
    at both ends (avoids slicing through the adjacent joint where the loop
    count goes wrong)."""
    head_pos = np.asarray(head_pos)
    tail_pos = np.asarray(tail_pos)
    axis = tail_pos - head_pos
    length = np.linalg.norm(axis)
    axis = axis / length
    ts = np.linspace(margin, 1 - margin, n_slices)
    out = []
    for t in ts:
        p = head_pos + t * length * axis
        perim, n_loops, dist = section_perimeter(mesh, origin=p, normal=axis, centerline_point=p)
        # Reject a slice where even the NEAREST loop is far from p -- means
        # this scan point is still inside the torso/pelvis and the only
        # available loop is the wrong body part's (the bug that gave
        # leg_circ=101cm the first time: t=0.15-0.20 near the hip returned
        # one loop at 10.9cm away, the pelvis, not the thigh).
        out.append(perim if (perim is not None and dist < 0.03) else np.nan)
    return np.array(out), ts, length


# Left thigh: LeftLeg (68, hip) -> LeftShin (69, knee).
LEFT_LEG, LEFT_SHIN = 68, 69
hip_pos = rest_bone_poses[0, LEFT_LEG, :3, 3].numpy()
knee_pos = rest_bone_poses[0, LEFT_SHIN, :3, 3].numpy()
thigh_profile, thigh_ts, thigh_len = limb_circumference_profile(mesh, hip_pos, knee_pos, 40)
valid_thigh = ~np.isnan(thigh_profile)
if valid_thigh.sum() < 5:
    raise SystemExit(f"CONTROL FAILED: too few clean thigh slices ({valid_thigh.sum()}/40)")
leg_circ = np.nanmax(thigh_profile) * M_TO_CM
print(f"leg_circ (left thigh max): {leg_circ:.2f} cm  (thigh length {thigh_len*M_TO_CM:.1f} cm)")

# Left wrist: LeftForeArm (14, elbow) -> LeftHand (15, wrist); thinnest part
# near the hand end, so scan the distal half only.
LEFT_FOREARM, LEFT_HAND = 14, 15
elbow_pos = rest_bone_poses[0, LEFT_FOREARM, :3, 3].numpy()
wrist_pos = rest_bone_poses[0, LEFT_HAND, :3, 3].numpy()
forearm_profile, forearm_ts, forearm_len = limb_circumference_profile(
    mesh, elbow_pos, wrist_pos, 40, margin=0.05
)
valid_forearm = ~np.isnan(forearm_profile)
if valid_forearm.sum() < 5:
    raise SystemExit(f"CONTROL FAILED: too few clean forearm slices ({valid_forearm.sum()}/40)")
# Thinnest part AT THE ROOT OF THE HAND = the min near the wrist end (last
# third of the scan), not the global min (which could be near the elbow on
# some phenotypes).
distal_third = forearm_ts > 0.66
distal_valid = valid_forearm & distal_third
if distal_valid.sum() < 3:
    raise SystemExit("CONTROL FAILED: too few clean slices in the distal forearm third for wrist")
wrist_circ = np.nanmin(forearm_profile[distal_valid]) * M_TO_CM
print(f"wrist (distal forearm min): {wrist_circ:.2f} cm  (forearm length {forearm_len*M_TO_CM:.1f} cm)")

if not (25 < leg_circ < 75):
    raise SystemExit(f"CONTROL FAILED: leg_circ {leg_circ:.2f} cm outside plausible adult range")
if not (10 < wrist_circ < 25):
    raise SystemExit(f"CONTROL FAILED: wrist {wrist_circ:.2f} cm outside plausible adult range")
print("CONTROL PASSED: leg_circ and wrist in plausible adult range")

print()
print("--- Summary (cm) ---")
for name, val in (("waist", waist_circ), ("bust", bust_circ), ("underbust", underbust_circ),
                   ("hips", hips_circ), ("leg_circ", leg_circ), ("wrist", wrist_circ)):
    print(f"  {name}: {val:.2f}")

# height: trivial (already have it from the Z-extent, no plane-cross-section
# needed) -- the top of the head and the bottom of the feet.
height_cm = (verts[:, 2].max() - verts[:, 2].min()) * M_TO_CM
print(f"  height: {height_cm:.2f}")

# --- Write the PARTIAL body_params.py-format YAML (7 of 26 fields) ---
#
# The other 19 fields (waist_back_width, back_width, hip_back_width, neck_w,
# waist_line, waist_over_bust_line, bust_line, arm_length, head_l,
# shoulder_w, armscye_depth, bust_points, hips_line, bum_points,
# crotch_hip_diff, vert_bust_line, hip_inclination, shoulder_incl,
# arm_pose_angle) need geodesic surface distances and/or named landmarks
# (bust points, nape of neck, collarbone ends, crotch) that a plain
# plane-cross-section can't give -- NOT computed here. Writing a complete
# anny-soma-x.yaml with those 19 as hand-guessed numbers would silently
# ship wrong patterns; leaving them absent makes the gap visible to
# whatever consumes this file next instead.
import yaml

partial = {
    "body": {
        "waist": round(float(waist_circ), 4),
        "bust": round(float(bust_circ), 4),
        "underbust": round(float(underbust_circ), 4),
        "hips": round(float(hips_circ), 4),
        "leg_circ": round(float(leg_circ), 4),
        "wrist": round(float(wrist_circ), 4),
        "height": round(float(height_cm), 4),
    }
}
with open("build/anny_export/anny-soma-x.partial.yaml", "w") as f:
    f.write("# PARTIAL -- 7 of 26 body_params.py fields. See\n")
    f.write("# measure_anny_circumferences.py's module docstring for which\n")
    f.write("# 19 fields are still missing and why (geodesic/landmark\n")
    f.write("# measurements, not computed by this script).\n")
    yaml.safe_dump(partial, f, default_flow_style=False)
print("wrote build/anny_export/anny-soma-x.partial.yaml (7/26 fields)")
