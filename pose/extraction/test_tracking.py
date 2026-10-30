"""Does the tracker actually follow one person when the detector reorders its output?

Reproduces the observed failure without needing DWPose: two people in frame, the detector returns
them in an order that flips from frame to frame, which is exactly what makes `kp[0]` jump.
"""
import numpy as np
from dwpose_extract_worker_tracked import seed_index, track_index, bbox_from, CONF, L_SH, R_SH


def person(cx, cy, shoulder_w, conf=0.9, jitter=0.0, rng=None):
    """One detection: 133 keypoints with plausible shoulders, the rest scattered around them."""
    rng = rng or np.random.default_rng(0)
    k = np.zeros((133, 2), np.float32)
    k[:] = [cx, cy]
    k += rng.normal(0, shoulder_w * 0.25, k.shape)
    k[L_SH] = [cx - shoulder_w / 2 + jitter, cy]
    k[R_SH] = [cx + shoulder_w / 2 + jitter, cy]
    s = np.full(133, conf, np.float32)
    return k, s


def run(flip_schedule, W=800, H=800):
    """Feed a two-person scene whose detection ORDER follows flip_schedule; return the centres the
    tracker accepted and the centres naive kp[0] would have accepted."""
    rng = np.random.default_rng(1)
    signer_x, other_x = 300.0, 620.0
    tracked, naive = [], []
    prev_mid = prev_wid = None
    for t, flipped in enumerate(flip_schedule):
        a = person(signer_x + 2.0 * np.sin(t / 3), 400, 150, jitter=0.0, rng=rng)   # the signer
        b = person(other_x, 420, 96, rng=rng)                                        # a bystander
        kp, sc = (np.stack([b[0], a[0]]), np.stack([b[1], a[1]])) if flipped else \
                 (np.stack([a[0], b[0]]), np.stack([a[1], b[1]]))
        i, _ = track_index(kp, sc, prev_mid, prev_wid, (W, H))
        mid = (kp[i][L_SH] + kp[i][R_SH]) / 2
        prev_mid, prev_wid = mid, float(np.linalg.norm(kp[i][L_SH] - kp[i][R_SH]))
        tracked.append(float(mid[0]))
        naive.append(float(((kp[0][L_SH] + kp[0][R_SH]) / 2)[0]))
    return np.array(tracked), np.array(naive)


def jumps(xs, thresh=100.0):
    return int((np.abs(np.diff(xs)) > thresh).sum())


sched = [False] * 8 + [True] * 6 + [False] * 5 + [True] * 4 + [False] * 7
tracked, naive = run(sched)
print(f"frames                    {len(sched)}   detector order flips {sum(1 for a,b in zip(sched, sched[1:]) if a!=b)} times")
print(f"naive kp[0]  identity jumps {jumps(naive):2d}   x range {naive.min():.0f}-{naive.max():.0f}")
print(f"tracked      identity jumps {jumps(tracked):2d}   x range {tracked.min():.0f}-{tracked.max():.0f}")
assert jumps(naive) >= 4, "the test scene does not reproduce the failure"
assert jumps(tracked) == 0, f"tracker still switches person ({jumps(tracked)} jumps)"
assert abs(tracked.mean() - 300) < 10, f"tracker locked onto the wrong person ({tracked.mean():.0f})"

# it should re-seed rather than drag the track when the scene genuinely cuts
rng = np.random.default_rng(2)
far = person(700.0, 400, 150, rng=rng)
i, reseeded = track_index(np.stack([far[0]]), np.stack([far[1]]),
                          np.array([300.0, 400.0]), 150.0, (800, 800))
assert reseeded, "a jump of 2.7 shoulder widths should have re-seeded the track"

# a real bbox, not the all-zero dummy the old worker wrote
k, s = person(300, 400, 150, rng=rng)
b = bbox_from(k, s)
assert b[2] > 0 and b[3] > 0, f"bbox has no extent: {b}"
print(f"re-seed on a >{1.6} shoulder-width jump: yes   bbox xywh = "
      f"[{b[0]:.0f} {b[1]:.0f} {b[2]:.0f} {b[3]:.0f}] (was all zeros)")

# a single-person scene must be untouched: same answer as the old code
one = person(400, 400, 150, rng=rng)
i, _ = track_index(np.stack([one[0]]), np.stack([one[1]]), None, None, (800, 800))
assert i == 0, "single-person clips must still take detection 0"
print("single-person clips unchanged (index 0), so MM-WLAuslan and the synthetic corpus are untouched")
print("\nALL CHECKS PASSED")
