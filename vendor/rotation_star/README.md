Vendored from `sinew-mocap/solve`, `core/rotation-star/c/` @ the commit cloned 2026-11-08
(upstream README: `V-Sekai-archive/contract-rotation-star` `main` @ `0c3830c`). MIT
License, copyright K. S. Ernest (iFire) Lee and contributors.

What this is: the "rotational-star" rotation solver -- a Lean-verified C port
(transcribed function-for-function from `lean/RotationStar` into `c/rotation_star.c`).
`rs_star` + `rs_star_kabsch` fit a full oriented 3-axis frame (as a 7-point "star":
origin, o+-x, o+-y, o+-z) via Kabsch SVD, rather than a single free vector -- used here
to compose the hip-abduction delta rotation for the ANNY-SOMA-X `_apart.obj` export
with a fully-determined twist, instead of `rs_rodrigues`'s free-vector alignment (tried
first, unshipped: it left the twist about the hip-to-ankle axis undetermined and the
resulting rotation crossed the legs through the midline instead of spreading them).

Before trusting this build after any refresh: `bash c/build.sh` runs the vendored test
against the Lean oracle (`tests/oracle/oracle.txt`) to <1e-12, including negative
controls that must fail the gate.

Files:
  rotation_star.h        -- the C API (see its own header comment for every function)
  librotation_star.dylib -- built from rotation_star.c via:
                             cc -shared -fPIC -O2 -o librotation_star.dylib rotation_star.c -lm
  LICENSE                -- upstream's MIT license, carried unchanged
