"""Export ANNY-SOMA-X (soma rig/topology, phenotypes=all, female) as a
GarmentCode-format body pair: anny-soma-x.obj (straight/rest) and
anny-soma-x_apart.obj (legs apart, for the leg-circumference measurement
plane search upstream's pattern_fitter.py expects).

Mirrors the layout of assets/bodies/mean_all.obj / mean_all_apart.obj in the
V-Sekai-fire/GarmentCode fork.

COORDINATE-FRAME FIX (iFire caught this): the model's `pose_parameterization`
default used elsewhere in this repo is "local-ref", which measures pose
*relative to a baked-in reference animation frame* (`reference_bone_orientations`),
NOT relative to the rest/bind pose. For this rig that reference frame is a
lying-down mocap pose, not standing -- so calling model(pose_parameters=None)
or an "identity" local-ref pose silently returns geometry in that lying
reference's pose, not rest_vertices' pose, even though both claim to share
the same axis convention. Confirmed empirically:
  - rest_vertices extent (Z-up, standing): [0.976, 0.385, 1.567] -- 1.567 on Z
    is clearly standing height.
  - "vertices" under local-ref with zero/identity delta: extent
    [1.571, 1.571, 0.313] -- the lying reference pose, NOT standing rest.
Fix: use pose_parameterization="world" and pass pose_parameters = the
rest_bone_poses themselves (verified byte-for-byte equal to rest_vertices,
max abs diff 2.4e-8) as the baseline, then compose the hip-abduction
rotation into THOSE world transforms directly, never through local-ref.
"""
import pathlib

import numpy as np
import torch
from anny import Anny

OUT_DIR = "build/anny_export"
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

model = Anny(rig="soma", topology="soma", pose_parameterization="local-ref", phenotypes="all")
model.eval()

n_bones = len(model.bone_labels)
assert n_bones == 78, n_bones

# gender=1.0 is female in ANNY's convention (operator note from anny-makehuman-clothes-fit skill)
with torch.no_grad():
    rest_out = model(phenotype_kwargs={"gender": 1.0})
rest_bone_poses = rest_out["rest_bone_poses"]           # [1, 78, 4, 4], Z-up, standing
verts_rest = rest_out["rest_vertices"][0].numpy().astype(np.float64)
faces = np.asarray(model.faces, dtype=np.int64)
print("rest verts", verts_rest.shape, "faces", faces.shape)

# CONTROL 0: world parameterization with pose_parameters = rest_bone_poses
# unchanged must reproduce rest_vertices exactly (same frame, zero deformation).
with torch.no_grad():
    identity_check = model(
        pose_parameters=rest_bone_poses, pose_parameterization="world",
        phenotype_kwargs={"gender": 1.0},
    )
identity_diff = np.abs(identity_check["vertices"][0].numpy() - verts_rest).max()
print(f"world-param identity control: max abs diff = {identity_diff:.2e}")
if identity_diff > 1e-4:
    raise SystemExit(f"CONTROL FAILED: world-param identity does not match rest_vertices "
                      f"(max abs diff {identity_diff:.2e})")
print("CONTROL PASSED: world-param pose=rest_bone_poses reproduces rest_vertices")

def descendants(bone_idx, parents):
    """All bones in the kinematic subtree rooted at bone_idx (inclusive)."""
    out = {bone_idx}
    changed = True
    while changed:
        changed = False
        for i, p in enumerate(parents):
            if p in out and i not in out:
                out.add(i)
                changed = True
    return out


# --- IK via the org's own rotation-star solver, not a hand-rolled axis guess ---
#
# SIGNPOST: github.com/sinew-mocap/solve, core/rotation-star/ -- "the rotational-star
# rotation solver", a Lean-verified C port (lean/RotationStar, transcribed function-for-
# function into c/rotation_star.c; its own test suite in c/build.sh checks the C against
# the Lean oracle to <1e-12 and ships negative controls that must fail the gate -- run it
# before trusting this build, not just once ever: `bash core/rotation-star/c/build.sh`).
#
# rs_rodrigues(a, b) (minimal rotation taking one FREE vector a onto another b) was tried
# first and UNSHIPPED: it only constrains the hip-to-ankle direction, leaving the twist
# about that direction free/arbitrary, and the resulting rotation blew straight through
# the midline (apart_gap went negative -- legs crossed, not spread).
#
# rs_star + rs_star_kabsch fit a full ORIENTED FRAME (the bone's 3 local axes, as a 7-point
# "star": origin, o+x, o-x, o+y, o-y, o+z, o-z) instead of one vector, via Kabsch SVD on the
# two stars' covariance. That fixes the twist exactly where the known target frame puts it,
# so the delta rotation is fully determined by the frame we hand it, not left to whichever
# rotation Rodrigues happens to compute between two under-constrained vectors.
#
# The target frame itself: rotate the hip's rest LOCAL-Y axis by a known angle (standard
# axis-angle, no solver needed for a known-axis-known-angle rotation -- that part is a
# formula, not a fit). Axis + sign (local-Y, same sign for both legs) measured empirically
# against THIS standing frame earlier in this file's history; see the delta-rotation
# control below, which fails loudly if that calibration ever drifts.
#
# For a multi-constraint pose (bend the knee too, hit a 3D target under joint limits) the
# right tool is core/lbfgsb (bounded L-BFGS-B, GPU) in the same repo -- not this. A single
# rigid single-joint abduction target is simple enough for the star-Kabsch solve below.
import ctypes

_rs_lib_path = pathlib.Path(__file__).parent / "vendor" / "rotation_star" / "librotation_star.dylib"
_rs = ctypes.CDLL(str(_rs_lib_path))
_rs.rs_star.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3
_rs.rs_star.restype = None
_rs.rs_star_kabsch.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3
_rs.rs_star_kabsch.restype = None


def rs_star(frame_3x3, origin=(0.0, 0.0, 0.0)):
    """The 7-point star of a 3x3 frame (row-major) at origin: o, o+-x, o+-y, o+-z."""
    r_arr = (ctypes.c_double * 9)(*frame_3x3.reshape(-1).tolist())
    o_arr = (ctypes.c_double * 3)(*origin)
    pts = (ctypes.c_double * 21)()
    _rs.rs_star(r_arr, o_arr, pts)
    return np.array(pts[:])


def rs_star_kabsch(tgt_star, src_star):
    """Rotation R (3x3, row-major) with tgt_star - mean ~= R (src_star - mean)."""
    tgt_arr = (ctypes.c_double * 21)(*tgt_star.tolist())
    src_arr = (ctypes.c_double * 21)(*src_star.tolist())
    out = (ctypes.c_double * 9)()
    _rs.rs_star_kabsch(tgt_arr, src_arr, out)
    return np.array(out[:]).reshape(3, 3)


def axis_angle(axis, angle_rad):
    """Standard Rodrigues axis-angle rotation about a KNOWN axis by a KNOWN angle
    -- a formula, not a fit, so no solver call is needed for this part."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


parents = model.bone_parents
pose_world = rest_bone_poses.clone()

LEFT_LEG, RIGHT_LEG = 68, 73
APART_RAD = np.radians(22.0)  # abduct each leg ~22 deg outward

for hip_idx, angle in ((LEFT_LEG, -APART_RAD), (RIGHT_LEG, -APART_RAD)):
    rest_R_np = rest_bone_poses[0, hip_idx, :3, :3].numpy()
    local_y_in_world = rest_R_np[:, 1]  # the bone's own rest local-Y axis, in world
    Ry = axis_angle(local_y_in_world, angle)
    target_R_np = Ry @ rest_R_np  # the known target FRAME (not just a target point)

    src_star = rs_star(rest_R_np)
    tgt_star = rs_star(target_R_np)
    delta_world_R = rs_star_kabsch(tgt_star, src_star)
    delta_world_R_t = torch.from_numpy(delta_world_R)
    pivot_t = rest_bone_poses[0, hip_idx, :3, 3]

    # DELTA-ROTATION CONTROL: the star-Kabsch solve must recover the SAME
    # rotation we asked for (Ry), not some other rotation that happens to
    # satisfy the frame fit loosely -- catches a transposed/row-vs-column
    # convention mismatch between this ctypes binding and the C ABI.
    delta_check = np.abs(delta_world_R - Ry).max()
    if delta_check > 1e-9:
        raise SystemExit(f"CONTROL FAILED: star-Kabsch delta rotation does not match "
                          f"the requested axis-angle Ry (max abs diff {delta_check:.2e})")

    # Rigidly rotate the WHOLE leg sub-chain (hip + shin + foot + toes) about
    # the pivot by delta_world_R -- this is the FK propagation that
    # pose_parameterization="world" does NOT do on its own (it takes raw
    # per-bone world transforms with no parent-to-child propagation).
    for bone_idx in descendants(hip_idx, parents):
        old_R = rest_bone_poses[0, bone_idx, :3, :3]
        old_t = rest_bone_poses[0, bone_idx, :3, 3]
        pose_world[0, bone_idx, :3, :3] = delta_world_R_t @ old_R
        pose_world[0, bone_idx, :3, 3] = delta_world_R_t @ (old_t - pivot_t) + pivot_t
print("CONTROL PASSED: star-Kabsch delta rotation matches the requested axis-angle")

with torch.no_grad():
    apart_out = model(
        pose_parameters=pose_world, pose_parameterization="world",
        phenotype_kwargs={"gender": 1.0},
    )
verts_apart = apart_out["vertices"][0].numpy().astype(np.float64)
print("apart verts", verts_apart.shape)

# PLANTED CONTROL: ankle separation must actually increase under the apart
# pose, measured on the LeftToeEnd/RightToeEnd bone positions in the SAME
# (rest_bone_poses-consistent) world frame used throughout this script.
LEFT_TOE, RIGHT_TOE = 72, 77
rest_toeL = rest_bone_poses[0, LEFT_TOE, :3, 3].numpy()
rest_toeR = rest_bone_poses[0, RIGHT_TOE, :3, 3].numpy()
apart_toeL = apart_out["bone_poses"][0, LEFT_TOE, :3, 3].numpy()
apart_toeR = apart_out["bone_poses"][0, RIGHT_TOE, :3, 3].numpy()
rest_gap = rest_toeL[0] - rest_toeR[0]
apart_gap = apart_toeL[0] - apart_toeR[0]
print(f"toe X-gap: rest={rest_gap:.4f}  apart={apart_gap:.4f}  delta={apart_gap - rest_gap:.4f}")
if not (apart_gap > rest_gap + 0.3):
    raise SystemExit(f"CONTROL FAILED: apart pose did not increase ankle separation "
                      f"(rest_gap={rest_gap:.4f}, apart_gap={apart_gap:.4f})")
print("CONTROL PASSED: ankle separation increased under the apart pose")

# CONTROL: height (Z extent) must stay standing-scale for BOTH meshes --
# catches a silent regression back to the lying reference frame.
for name, v in (("rest", verts_rest), ("apart", verts_apart)):
    z_extent = v[:, 2].max() - v[:, 2].min()
    print(f"{name} Z-extent (height): {z_extent:.3f} m")
    if not (1.3 < z_extent < 2.0):
        raise SystemExit(f"CONTROL FAILED: {name} mesh Z-extent {z_extent:.3f} m is not "
                          f"standing-human scale -- wrong coordinate frame")
print("CONTROL PASSED: both meshes are standing-scale on Z")


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


write_obj(f"{OUT_DIR}/anny-soma-x.obj", verts_rest, faces)
write_obj(f"{OUT_DIR}/anny-soma-x_apart.obj", verts_apart, faces)

np.savez(f"{OUT_DIR}/anny-soma-x_rest.npz", verts=verts_rest, faces=faces)
np.savez(f"{OUT_DIR}/anny-soma-x_apart.npz", verts=verts_apart, faces=faces)

print("wrote", OUT_DIR)
