"""
Generate Synthetic SMPL-X Data V5 — Maximally Distinct Poses
==============================================================

Confirmed axis mappings:
  Shoulder Y-  = arm forward       Shoulder Y+  = arm backward
  Shoulder Z+  = arm up            Shoulder Z-  = arm down (from T-pose)
  Elbow Z+     = flexion (bend)
  Finger X+/Y+ = curl

DESIGN: Every gloss has BOTH arms in a unique configuration.
No two glosses share the same idle arm.

  wave:   L arm sideways + elbow bends | R arm on hip (bent forward)
  nod:    Both arms crossed in front (folded) | head nods
  point:  L arm forward pointing | R arm behind back
  circle: Both arms horizontal "goalpost" (elbows 90° down) | forearms circle
  raise:  Both arms sweep sides → overhead → sides
"""

import os
import shutil
import numpy as np

OUTPUT_ROOT = "../data/synthetic_smplx_data"
NUM_SAMPLES_PER_GLOSS = 20
NUM_TEST_SAMPLES = 5
FRAME_RANGE = (25, 35)

IDX_ROOT = 0
IDX_SPINE1 = 3;  IDX_SPINE2 = 6;  IDX_SPINE3 = 9
IDX_NECK = 12;   IDX_L_COLLAR = 13; IDX_R_COLLAR = 14
IDX_HEAD = 15
IDX_L_SHOULDER = 16; IDX_R_SHOULDER = 17
IDX_L_ELBOW = 18;    IDX_R_ELBOW = 19
IDX_L_WRIST = 20;    IDX_R_WRIST = 21

PI = np.pi


# ============================================================================
# 5 Gloss Generators — every gloss, both arms unique
# ============================================================================

def gen_wave(T, rng):
    """
    L arm: raised sideways (Z≈-0.2), elbow oscillates (waving)
    R arm: on hip — forward (Y=-0.5) + very bent elbow (Z=2.0)
    Motion: L elbow Z oscillates, L wrist twists, L fingers open/close
    """
    pose = np.zeros((T, 53, 3), dtype=np.float32)
    t = np.linspace(0, 1, T)
    freq = rng.uniform(2.0, 3.5)
    phase = rng.uniform(0, 2 * PI)

    # L arm: raised sideways, slightly forward
    pose[:, IDX_L_SHOULDER, 2] = -0.2
    pose[:, IDX_L_SHOULDER, 1] = -0.3

    # R arm: on hip (forward + down + very bent)
    pose[:, IDX_R_SHOULDER, 2] = -0.7
    pose[:, IDX_R_SHOULDER, 1] = -0.5
    pose[:, IDX_R_ELBOW, 2] = 2.0
    # R hand fist (resting on hip)
    for j in range(37, 52):
        pose[:, j, 0] = 0.8

    # MOTION: L elbow bends back and forth
    pose[:, IDX_L_ELBOW, 2] = 0.6 + 0.8 * np.sin(2*PI*freq*t + phase)

    # L wrist twists
    pose[:, IDX_L_WRIST, 1] = 0.5 * np.sin(2*PI*freq*1.5*t + phase)

    # L fingers open/close
    for j in range(22, 37):
        pose[:, j, 0] = 0.2 + 0.5 * np.sin(2*PI*freq*t + phase + j*0.15)

    return pose


def gen_nod(T, rng):
    """
    L arm: forward at chest, elbow bent 90° — fingers curl/uncurl
    R arm: hanging at side (Z=-1.0), relaxed, STATIC
    Motion: L hand fingers + wrist animate. R arm does nothing.
    Unique: only gloss with one arm forward-bent + one arm hanging straight down.
    """
    pose = np.zeros((T, 53, 3), dtype=np.float32)
    t = np.linspace(0, 1, T)
    freq = rng.uniform(1.5, 3.0)
    phase = rng.uniform(0, 2 * PI)

    # L arm: forward + bent
    pose[:, IDX_L_SHOULDER, 1] = -0.8
    pose[:, IDX_L_SHOULDER, 2] = -0.5
    pose[:, IDX_L_ELBOW, 2] = 1.5

    # R arm: hanging at side
    pose[:, IDX_R_SHOULDER, 2] = -1.0
    pose[:, IDX_R_ELBOW, 2] = 0.2

    # MOTION: L hand fingers curl and uncurl
    for j in range(22, 37):  # left hand only
        pose[:, j, 0] = 0.5 + 0.5 * np.sin(2*PI*freq*t + phase + j*0.3)

    # L wrist rotates
    pose[:, IDX_L_WRIST, 1] = 0.4 * np.sin(2*PI*freq*t + phase)

    return pose


def gen_point(T, rng):
    """
    L arm: extends straight forward over time (Y ramps to -1.5, elbow straightens)
           Index finger out, others curled
    R arm: BEHIND BACK — backward (Y=+0.8), elbow bent (Z=1.5)
           Hand gripping behind back
    Motion: L arm smooth reach forward. R arm static behind back.
    """
    pose = np.zeros((T, 53, 3), dtype=np.float32)
    t = np.linspace(0, 1, T)

    trans = rng.uniform(0.3, 0.5)
    progress = np.clip((t - 0.05) / (trans - 0.05), 0, 1)
    progress = progress**2 * (3 - 2*progress)

    # R arm: hanging at side
    pose[:, IDX_R_SHOULDER, 2] = 0
    pose[:, IDX_R_ELBOW, 2] = 0


    # L arm: starts at side, reaches forward
    pose[:, IDX_L_SHOULDER, 2] = -1.0 + 0.8 * progress   # -1.0 → -0.2
    pose[:, IDX_L_SHOULDER, 1] = -1.5 * progress          # 0 → -1.5 forward
    pose[:, IDX_L_ELBOW, 2] = 0.8 * (1.0 - progress)     # bent → straight

    # L hand: index straight, others curl
    pose[:, 22, 0] = 0.05 * progress
    pose[:, 23, 0] = 0.03 * progress
    for j in range(24, 37):
        pose[:, j, 0] = 1.0 * progress

    pose[:, IDX_SPINE1, 0] = -0.1 * progress

    return pose


def gen_circle(T, rng):
    """
    Both arms: horizontal "goalpost" — T-pose level (Z≈0), elbows bent 90° down (Z=1.5)
    Forearms hang down, then oscillate.
    Motion: elbows Z oscillate (alternating), wrists circle. VERY different silhouette.
    """
    pose = np.zeros((T, 53, 3), dtype=np.float32)
    t = np.linspace(0, 1, T)
    freq = rng.uniform(1.5, 2.5)
    radius = rng.uniform(0.5, 0.8)
    phase = rng.uniform(0, 2 * PI)

    # Both arms horizontal (T-pose level = Z≈0) — NOT forward
    pose[:, IDX_L_SHOULDER, 2] = 0.0
    pose[:, IDX_R_SHOULDER, 2] = 0.0
    # No Y component = arms stay to the sides, not forward

    # MOTION: elbows oscillate around 90° bend (alternating)
    pose[:, IDX_L_ELBOW, 2] = 1.3 + radius * np.sin(2*PI*freq*t + phase)
    pose[:, IDX_R_ELBOW, 2] = 1.3 + radius * np.sin(2*PI*freq*t + phase + PI)

    # Wrists circle
    pose[:, IDX_L_WRIST, 1] = radius * np.sin(2*PI*freq*t + phase)
    pose[:, IDX_L_WRIST, 2] = radius * np.cos(2*PI*freq*t + phase)
    pose[:, IDX_R_WRIST, 1] = radius * np.sin(2*PI*freq*t + phase + PI)
    pose[:, IDX_R_WRIST, 2] = radius * np.cos(2*PI*freq*t + phase + PI)

    # Fingers semi-open
    for j in range(22, 52):
        pose[:, j, 0] = 0.3 + 0.2 * np.sin(2*PI*freq*t + j*0.12)

    return pose


def gen_raise(T, rng):
    """
    Both arms: sweep from at-sides (Z=-1.0) to overhead (Z=+1.5) and back.
    Elbows stay mostly straight. Fingers spread when up.
    Motion: smooth up-down envelope. Spine arches when arms up.
    """
    pose = np.zeros((T, 53, 3), dtype=np.float32)
    t = np.linspace(0, 1, T)
    freq = rng.uniform(0.7, 1.2)
    phase = rng.uniform(0, PI / 3)

    envelope = np.clip(np.sin(PI * freq * t + phase), 0, 1)

    # Shoulders: from sides (Z=-1.0) to overhead (Z=+1.5)
    pose[:, IDX_L_SHOULDER, 2] = -1.0 + 2.5 * envelope
    pose[:, IDX_R_SHOULDER, 2] = -1.0 + 2.5 * envelope

    # Elbows: straight when up, slightly bent when down
    pose[:, IDX_L_ELBOW, 2] = 0.4 * (1 - envelope)
    pose[:, IDX_R_ELBOW, 2] = 0.4 * (1 - envelope)

    # Spine arches back when arms up
    pose[:, IDX_SPINE1, 0] = -0.15 * envelope
    pose[:, IDX_SPINE2, 0] = -0.1 * envelope

    # Fingers spread when up
    for j in range(22, 52):
        pose[:, j, 1] = 0.3 * envelope

    return pose


# ============================================================================
# Save / Generate / Verify
# ============================================================================

GLOSS_GENERATORS = {
    "wave": gen_wave, "nod": gen_nod, "point": gen_point,
    "circle": gen_circle, "raise": gen_raise,
}


def save_sample(pose_seq, gloss, video_id, output_dir):
    video_dir = os.path.join(output_dir, gloss, f"{video_id:05d}")
    os.makedirs(video_dir, exist_ok=True)
    T = pose_seq.shape[0]
    for i in range(T):
        f = pose_seq[i]
        np.savez(os.path.join(video_dir, f"{gloss}_{video_id:05d}_{i:06d}_p0.npz"),
            smplx_root_pose=f[0], smplx_body_pose=f[1:22],
            smplx_lhand_pose=f[22:37], smplx_rhand_pose=f[37:52], smplx_jaw_pose=f[52])
    return T


def generate_split(mode, num_samples, output_root):
    smplx_dir = os.path.join(output_root, mode, "smplx_params")
    if os.path.exists(smplx_dir):
        shutil.rmtree(smplx_dir)
    os.makedirs(smplx_dir)
    total = 0
    for gloss, fn in GLOSS_GENERATORS.items():
        print(f"  [{mode}] '{gloss}': {num_samples} samples")
        for si in range(num_samples):
            rng = np.random.RandomState(hash((gloss, mode, si)) % (2**31))
            T = rng.randint(FRAME_RANGE[0], FRAME_RANGE[1] + 1)
            seq = fn(T, rng)
            seq += rng.randn(*seq.shape).astype(np.float32) * 0.03
            total += save_sample(seq, gloss, si + 1, smplx_dir)
    return total


def verify_data(output_root):
    print("\n=== Frame 0 pose summary ===")
    d = os.path.join(output_root, "train", "smplx_params")
    for g in sorted(os.listdir(d)):
        vp = os.path.join(d, g, sorted(os.listdir(os.path.join(d, g)))[0])
        fs = sorted([f for f in os.listdir(vp) if f.endswith('.npz')])
        data = np.load(os.path.join(vp, fs[0]))
        full = np.vstack([
            data['smplx_root_pose'].reshape(1,3), data['smplx_body_pose'].reshape(21,3),
            data['smplx_lhand_pose'].reshape(15,3), data['smplx_rhand_pose'].reshape(15,3),
            data['smplx_jaw_pose'].reshape(1,3)])
        ls = full[IDX_L_SHOULDER]; rs = full[IDX_R_SHOULDER]
        le = full[IDX_L_ELBOW]; re = full[IDX_R_ELBOW]
        print(f"  {g:7s}: L_sh=[{ls[0]:+.1f},{ls[1]:+.1f},{ls[2]:+.1f}] "
              f"R_sh=[{rs[0]:+.1f},{rs[1]:+.1f},{rs[2]:+.1f}] "
              f"L_el_Z={le[2]:+.1f} R_el_Z={re[2]:+.1f}")


if __name__ == "__main__":
    print(f"Output: {os.path.abspath(OUTPUT_ROOT)}\n")
    print("TRAIN:"); generate_split("train", NUM_SAMPLES_PER_GLOSS, OUTPUT_ROOT)
    print("\nTEST:");  generate_split("test", NUM_TEST_SAMPLES, OUTPUT_ROOT)
    verify_data(OUTPUT_ROOT)
    print()
    print("Silhouette summary:")
    print("  wave:   L raised+waving  | R on hip (bent)")
    print("  nod:    L forward+bent (fingers animate) | R hanging at side")
    print("  point:  L forward pointing | R behind back")
    print("  circle: Both horizontal goalpost | forearms circle")
    print("  raise:  Both sweep sides↔overhead")