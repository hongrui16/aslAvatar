"""
Local renderer for ASL Signbank SMPL-X outputs.

Directory structure expected:
  asl_signbank/
  ├── smplx_params/
  │   └── {GLOSS}/
  │       ├── {GLOSS}_{FRAME:06d}_p0.obj
  │       ├── {GLOSS}_{FRAME:06d}_p0.npz
  │       ...
  └── videos/
      └── {GLOSS}.mp4

Usage:
  # Render one gloss (auto-extracts video frames as background)
  python render_local.py --root ./asl_signbank --gloss AMAZING

  # Render multiple glosses
  python render_local.py --root ./asl_signbank --gloss AMAZING HELLO THANK-YOU

  # Render ALL glosses found in smplx_params/
  python render_local.py --root ./asl_signbank --all

  # White background (skip video frame extraction)
  python render_local.py --root ./asl_signbank --gloss AMAZING --no_bg

  # Custom output dir
  python render_local.py --root ./asl_signbank --gloss AMAZING --out_dir ./my_renders


  # 视频帧背景（默认）
python render_local.py --root ./asl_signbank --gloss AMAZING

# 白色背景
python render_local.py --root ./asl_signbank --gloss AMAZING --no_bg

# 手动调透明度（1.0 = 完全不透明）
python render_local.py --root ./asl_signbank --gloss AMAZING --alpha 0.95

"""

import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import re
import argparse
import glob
import numpy as np
import cv2
import trimesh
import pyrender
import sys

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from utils.renders import render_mesh

# ─── I/O helpers ──────────────────────────────────────────────────────

def load_obj(obj_path):
    mesh = trimesh.load(obj_path, process=False)
    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces, dtype=np.int32)


def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {
        "focal": data["focal"],        # [fx, fy]
        "princpt": data["princpt"],    # [cx, cy]
        "cam_trans": data.get("cam_trans", None),
    }


def parse_frame_idx(filename, gloss):
    """
    Extract frame index from filename like AMAZING_000004_p0.obj
    Returns int or None.
    """
    stem = os.path.splitext(filename)[0]
    pattern = re.escape(gloss) + r"_(\d+)_p\d+"
    m = re.match(pattern, stem)
    if m:
        return int(m.group(1))
    return None



# ─── Per-gloss pipeline ──────────────────────────────────────────────

def render_gloss(root, gloss, out_dir, use_bg=True, alpha=0.8):
    """Render all keyframes for a single gloss."""
    param_dir = os.path.join(root, "smplx_params", gloss)
    video_path = os.path.join(root, "videos", f"{gloss}.mp4")

    if not os.path.isdir(param_dir):
        print(f"[SKIP] Param dir not found: {param_dir}")
        return

    obj_files = sorted(glob.glob(os.path.join(param_dir, "*.obj")))
    if not obj_files:
        print(f"[SKIP] No .obj files in {param_dir}")
        return

    gloss_out_dir = os.path.join(out_dir, gloss)
    os.makedirs(gloss_out_dir, exist_ok=True)

    # Open video once, seek per frame
    has_video = os.path.isfile(video_path) and use_bg
    if use_bg and not has_video:
        print(f"[WARN] Video not found: {video_path}, using white background")

    cap = None
    if has_video:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open {video_path}, using white background")
            cap = None

    rendered_count = 0
    for obj_path in obj_files:
        stem = os.path.splitext(os.path.basename(obj_path))[0]
        npz_path = os.path.join(param_dir, stem + ".npz")

        if not os.path.isfile(npz_path):
            print(f"[WARN] Missing .npz for {obj_path}, skipping")
            continue

        frame_idx = parse_frame_idx(os.path.basename(obj_path), gloss)

        # Load mesh + camera params
        vertices, faces = load_obj(obj_path)
        cam = load_npz(npz_path)

        # Get background frame from video
        bg = None
        if cap is not None and frame_idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if ret:
                bg = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if bg is None:
            # Fallback: white canvas sized from camera principal point
            cx, cy = cam["princpt"]
            w = int(cx * 2) if cx > 100 else 960
            h = int(cy * 2) if cy > 100 else 540
            bg = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Render overlay
        rendered = render_mesh(bg, vertices, faces, cam, alpha=alpha)

        out_path = os.path.join(gloss_out_dir, f"{stem}_render.png")
        cv2.imwrite(out_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        rendered_count += 1

    if cap is not None:
        cap.release()

    print(f"[{gloss}] Rendered {rendered_count} frames -> {gloss_out_dir}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASL Signbank SMPL-X local renderer")
    parser.add_argument("--root", type=str, required=True,
                        help="Root dir containing smplx_params/ and videos/")
    parser.add_argument("--gloss", type=str, nargs="+", default=None,
                        help="One or more gloss names to render")
    parser.add_argument("--all", action="store_true",
                        help="Render all glosses in smplx_params/")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: {root}/rendered)")
    parser.add_argument("--no_bg", action="store_true",
                        help="White background instead of video frames")
    parser.add_argument("--alpha", type=float, default=0.92,
                        help="Mesh overlay transparency [0-1]")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.root, "rendered")

    if args.all:
        param_root = os.path.join(args.root, "smplx_params")
        glosses = sorted([
            d for d in os.listdir(param_root)
            if os.path.isdir(os.path.join(param_root, d))
        ])
    elif args.gloss:
        glosses = args.gloss
    else:
        parser.error("Specify --gloss GLOSS_NAME or --all")
        return

    print(f"Rendering {len(glosses)} gloss(es)...")
    for g in glosses:
        render_gloss(args.root, g, out_dir, use_bg=not args.no_bg, alpha=args.alpha)

    print(f"\nDone! All outputs in: {out_dir}")


if __name__ == "__main__":
    main()