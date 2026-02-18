"""
SMPL-X Joint Axis Diagnostic
==============================

Problem: We don't know which axis-angle component causes elbow FLEXION vs twist.

This script generates 7 single-frame test poses:
  test_00_neutral    : Natural T-pose (zero everything)
  test_01_elbow_X    : Left elbow axis-angle X = 1.57 (90°)
  test_02_elbow_Y    : Left elbow axis-angle Y = 1.57
  test_03_elbow_Z    : Left elbow axis-angle Z = 1.57
  test_04_shoulder_X : Left shoulder axis-angle X = -1.57 (arm forward?)
  test_05_shoulder_Y : Left shoulder axis-angle Y = 1.57
  test_06_shoulder_Z : Left shoulder axis-angle Z = 1.57 (arm down from T-pose?)

Render all 7 frames and check:
  - Which elbow test (01/02/03) actually BENDS the arm?
  - Which shoulder test puts the arm where expected?

Then we know exactly which axis does what.

Usage:
  python diagnostic_smplx_axes.py
  python render_synthetic.py --root ./diagnostic_data --mode train --gif
"""

import os
import shutil
import numpy as np

OUTPUT_ROOT = "../data/diagnostic_data"
# Same joint indices
IDX_L_SHOULDER = 16
IDX_R_SHOULDER = 17
IDX_L_ELBOW = 18
IDX_R_ELBOW = 19
IDX_L_WRIST = 20
IDX_R_WRIST = 21

HALF_PI = np.pi / 2  # 90°


def make_zero_pose():
    """Single frame, all zeros (T-pose)."""
    return np.zeros((1, 53, 3), dtype=np.float32)


def save_test(pose, name, output_dir):
    """Save a single-frame pose as a 'video' with 1 frame."""
    video_dir = os.path.join(output_dir, name, "00001")
    os.makedirs(video_dir, exist_ok=True)
    f = pose[0]
    np.savez(os.path.join(video_dir, f"{name}_00001_000000_p0.npz"),
        smplx_root_pose=f[0],
        smplx_body_pose=f[1:22],
        smplx_lhand_pose=f[22:37],
        smplx_rhand_pose=f[37:52],
        smplx_jaw_pose=f[52],
    )


def main():
    smplx_dir = os.path.join(OUTPUT_ROOT, "train", "smplx_params")
    if os.path.exists(smplx_dir):
        shutil.rmtree(smplx_dir)
    os.makedirs(smplx_dir)

    # Test 0: Neutral T-pose (baseline)
    pose = make_zero_pose()
    save_test(pose, "test00_neutral", smplx_dir)
    print("test00_neutral: All zeros (T-pose)")

    # Test 1-3: Left elbow on each axis
    for axis, axis_name in enumerate(["X", "Y", "Z"]):
        pose = make_zero_pose()
        pose[0, IDX_L_ELBOW, axis] = HALF_PI
        save_test(pose, f"test0{axis+1}_elbow_{axis_name}", smplx_dir)
        print(f"test0{axis+1}_elbow_{axis_name}: Left elbow [{['1.57' if i==axis else '0' for i in range(3)]}]")

    # Test 4-6: Left shoulder on each axis
    for axis, axis_name in enumerate(["X", "Y", "Z"]):
        pose = make_zero_pose()
        pose[0, IDX_L_SHOULDER, axis] = HALF_PI
        save_test(pose, f"test0{axis+4}_shoulder_{axis_name}", smplx_dir)
        print(f"test0{axis+4}_shoulder_{axis_name}: Left shoulder [{['1.57' if i==axis else '0' for i in range(3)]}]")

    # Test 7-9: Left shoulder NEGATIVE on each axis
    for axis, axis_name in enumerate(["X", "Y", "Z"]):
        pose = make_zero_pose()
        pose[0, IDX_L_SHOULDER, axis] = -HALF_PI
        save_test(pose, f"test{axis+7:02d}_shoulder_neg{axis_name}", smplx_dir)
        print(f"test{axis+7:02d}_shoulder_neg{axis_name}: Left shoulder [{['-1.57' if i==axis else '0' for i in range(3)]}]")

    # Test 10-12: Left wrist on each axis
    for axis, axis_name in enumerate(["X", "Y", "Z"]):
        pose = make_zero_pose()
        pose[0, IDX_L_WRIST, axis] = HALF_PI
        save_test(pose, f"test{axis+10:02d}_wrist_{axis_name}", smplx_dir)
        print(f"test{axis+10:02d}_wrist_{axis_name}: Left wrist [{['1.57' if i==axis else '0' for i in range(3)]}]")

    # Test 13: Fingers - curl all left hand joints on X
    pose = make_zero_pose()
    for j in range(22, 37):
        pose[0, j, 0] = 1.0
    save_test(pose, "test13_lhand_curl_X", smplx_dir)
    print("test13_lhand_curl_X: All left hand joints X=1.0")

    # Test 14: Fingers - curl all left hand joints on Y
    pose = make_zero_pose()
    for j in range(22, 37):
        pose[0, j, 1] = 1.0
    save_test(pose, "test14_lhand_curl_Y", smplx_dir)
    print("test14_lhand_curl_Y: All left hand joints Y=1.0")

    # Test 15: Fingers - curl all left hand joints on Z
    pose = make_zero_pose()
    for j in range(22, 37):
        pose[0, j, 2] = 1.0
    save_test(pose, "test15_lhand_curl_Z", smplx_dir)
    print("test15_lhand_curl_Z: All left hand joints Z=1.0")

    print(f"\nSaved to: {os.path.abspath(smplx_dir)}")
    print(f"Total: 16 test poses")
    print(f"\nRender with:")
    print(f"  python render_synthetic.py --root {OUTPUT_ROOT} --mode train --all_samples")
    print(f"\nThen tell me:")
    print(f"  1. Which elbow test (01/02/03) BENDS the arm?")
    print(f"  2. Which shoulder test puts arm DOWN from T-pose?")
    print(f"  3. Which hand test CURLS the fingers?")


if __name__ == "__main__":
    main()
