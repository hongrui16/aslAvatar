"""
SignBank SMPL-X Inference Script (params only)

Generate sign language motion from gloss, save SMPL-X parameters (.npz) per frame.

Output:
    output_dir/
        GLOSS_NAME/
            GLOSS_NAME_000000_p0.npz
            GLOSS_NAME_000001_p0.npz
            ...

Usage:
    python inference_signbank_smplx.py \
        --checkpoint path/to/best_model.pt \
        --glosses AMAZING HELLO THANK-YOU

    python inference_signbank_smplx.py \
        --checkpoint path/to/best_model.pt \
        --from_dataset --num_glosses 20
"""

import os
import argparse
import random
from typing import List, Dict, Optional

import torch
import numpy as np
from tqdm import tqdm

from aslAvatarModel import ASLAvatarModel
from conflg import SignBank_SMPLX_Config


# =============================================================================
# SMPL-X Parameter Layout (159 dims)
#   root_pose(3) + body_pose(63) + lhand(45) + rhand(45) + jaw(3)
# =============================================================================

PARAM_SLICES = {
    'smplx_root_pose':  (0,   3),
    'smplx_body_pose':  (3,   66),
    'smplx_lhand_pose': (66,  111),
    'smplx_rhand_pose': (111, 156),
    'smplx_jaw_pose':   (156, 159),
}


def split_params(flat: np.ndarray) -> Dict[str, np.ndarray]:
    """Split (159,) vector into named SMPL-X components."""
    return {name: flat[s:e].copy() for name, (s, e) in PARAM_SLICES.items()}


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda'):
    cfg = SignBank_SMPLX_Config()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'config' in ckpt:
        for k, v in ckpt['config'].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    model = ASLAvatarModel(cfg)
    model_state = ckpt.get('model_state_dict', ckpt)
    cur = model.state_dict()
    loaded = 0
    for k in model_state:
        if k in cur and cur[k].shape == model_state[k].shape:
            cur[k] = model_state[k]
            loaded += 1
    model.load_state_dict(cur, strict=False)
    model.to(device).eval()

    print(f"Loaded: {checkpoint_path}  (epoch {ckpt.get('epoch','?')}, {loaded} keys)")
    return model, cfg


# =============================================================================
# Generation
# =============================================================================

@torch.no_grad()
def generate_from_gloss(model, gloss: str, seq_len: int, input_dim: int, device: str):
    """Generate (T, 159) SMPL-X params from gloss condition."""
    dummy = torch.zeros(1, seq_len, input_dim, device=device)
    pad   = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    out, _, _ = model(dummy, [gloss], pad)
    return out.squeeze(0).cpu().numpy()


# =============================================================================
# Save .npz (matching your extraction pipeline format)
# =============================================================================

def save_frame_npz(frame_params: Dict[str, np.ndarray], save_path: str):
    dump = {
        'smplx_root_pose':  frame_params['smplx_root_pose'].reshape(3,).astype(np.float32),
        'smplx_body_pose':  frame_params['smplx_body_pose'].reshape(21, 3).astype(np.float32),
        'smplx_lhand_pose': frame_params['smplx_lhand_pose'].reshape(15, 3).astype(np.float32),
        'smplx_rhand_pose': frame_params['smplx_rhand_pose'].reshape(15, 3).astype(np.float32),
        'smplx_jaw_pose':   frame_params['smplx_jaw_pose'].reshape(3,).astype(np.float32),
        'smplx_shape':      np.zeros(10, dtype=np.float32),
        'smplx_expr':       np.zeros(10, dtype=np.float32),
        'cam_trans':         np.zeros(3, dtype=np.float32),
    }
    np.savez(save_path, **dump)


# =============================================================================
# Process one gloss
# =============================================================================

def process_gloss(model, gloss, output_dir, seq_len, input_dim, device):
    motion = generate_from_gloss(model, gloss, seq_len, input_dim, device)  # (T, 159)
    T = motion.shape[0]

    gloss_dir = os.path.join(output_dir, gloss)
    os.makedirs(gloss_dir, exist_ok=True)

    for t in range(T):
        params = split_params(motion[t])
        npz_path = os.path.join(gloss_dir, f"{gloss}_{t:06d}_p0.npz")
        save_frame_npz(params, npz_path)

    return T


# =============================================================================
# Gloss discovery from dataset dir
# =============================================================================

def get_glosses_from_dataset(root_dir: str, num_glosses: Optional[int] = None) -> List[str]:
    if not os.path.isdir(root_dir):
        print(f"WARNING: not found: {root_dir}")
        return []
    glosses = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    if num_glosses and num_glosses < len(glosses):
        glosses = random.sample(glosses, num_glosses)
    return glosses


# =============================================================================
# Main
# =============================================================================

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    model, cfg = load_model(args.checkpoint, device)
    input_dim = cfg.INPUT_DIM
    seq_len = args.seq_len if args.seq_len else cfg.TARGET_SEQ_LEN

    # Determine glosses
    if args.glosses:
        glosses = args.glosses
    elif args.from_dataset:
        root = args.dataset_dir or getattr(cfg, 'ROOT_DIR', '')
        glosses = get_glosses_from_dataset(root, args.num_glosses)
    else:
        print("ERROR: provide --glosses or --from_dataset")
        return

    print(f"Generating {len(glosses)} glosses, {seq_len} frames each")
    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    for gloss in tqdm(glosses, desc="Generating"):
        total += process_gloss(model, gloss, args.output_dir, seq_len, input_dim, device)

    print(f"\nDone! {len(glosses)} glosses, {total} frames -> {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SignBank SMPL-X Inference (params only)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--glosses", type=str, nargs='+', default=None, help="e.g. --glosses AMAZING HELLO")
    p.add_argument("--from_dataset", action="store_true")
    p.add_argument("--dataset_dir", type=str, default=None)
    p.add_argument("--num_glosses", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="./inference_results")
    p.add_argument("--seq_len", type=int, default=None, help="Override frame count (default: cfg.TARGET_SEQ_LEN=48)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
