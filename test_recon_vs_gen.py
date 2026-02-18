"""
Diagnostic: Compare reconstruction vs generation temporal variation.

Tests whether the decoder produces frame-to-frame motion differences.
If reconstruction also has no variation, the problem is architectural.
If only generation has no variation, the problem is latent space coverage.

Usage:
    python test_recon_vs_gen.py \
        --checkpoint path/to/best_model.pt \
        --dataset_name WLASL_SMPLX

    python test_recon_vs_gen.py \
        --checkpoint path/to/best_model.pt \
        --dataset_name SignBank_SMPLX

    # With mesh rendering
    python test_recon_vs_gen.py \
        --checkpoint path/to/best_model.pt \
        --dataset_name WLASL_SMPLX \
        --render_mesh
"""

import os
import sys
import argparse
import random

import torch
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aslAvatarModel import ASLAvatarModel
from config import SignBank_SMPLX_Config, WLASL_SMPLX_Config


def load_model(cfg, checkpoint_path, device):
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
    print(f"Loaded: {checkpoint_path}  (epoch {ckpt.get('epoch', '?')}, {loaded} keys)")
    return model, cfg


def load_dataset(cfg, dataset_name, mode='train'):
    if dataset_name == "WLASL_SMPLX":
        from dataloader.WLASLSMPLXDataset import WLASLSMPLXDataset
        return WLASLSMPLXDataset(mode=mode, cfg=cfg)
    elif dataset_name == "SignBank_SMPLX":
        from dataloader.SignBankSMPLXDataset import SignBankSMPLXDataset
        return SignBankSMPLXDataset(mode=mode, cfg=cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def analyze_temporal_variation(seq, label, length=None):
    """Compute frame-to-frame stats for a (T, D) numpy array."""
    if length is not None:
        seq = seq[:length]
    if len(seq) < 2:
        return {"mean_diff": 0, "max_diff": 0, "std_per_frame": 0, "num_frames": len(seq)}

    diffs = np.diff(seq, axis=0)
    return {
        "label": label,
        "num_frames": len(seq),
        "mean_diff": float(np.abs(diffs).mean()),
        "max_diff": float(np.abs(diffs).max()),
        "std_per_frame": float(seq.std(axis=0).mean()),
        "frame_norms": [float(np.linalg.norm(seq[t])) for t in range(min(5, len(seq)))],
    }


def print_stats(name, stats):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Gloss:          {stats['label']}")
    print(f"  Num frames:     {stats['num_frames']}")
    print(f"  Mean frame diff: {stats['mean_diff']:.6f}")
    print(f"  Max frame diff:  {stats['max_diff']:.6f}")
    print(f"  Std per frame:   {stats['std_per_frame']:.6f}")
    print(f"  First 5 frame norms: {['%.4f' % n for n in stats['frame_norms']]}")


@torch.no_grad()
def run_diagnosis(model, dataset, device, num_samples=5, render=False, output_dir=None):
    """Run reconstruction and generation tests on a few samples."""

    seq_len = dataset.target_seq_len
    input_dim = dataset.input_dim

    # Pick random samples from different glosses
    all_glosses = list(dataset.gloss_to_idx.keys())
    selected_glosses = random.sample(all_glosses, min(num_samples, len(all_glosses)))

    # Find one sample per selected gloss
    sample_indices = []
    for gloss in selected_glosses:
        for i, (g, _) in enumerate(dataset.data_list):
            if g == gloss:
                sample_indices.append(i)
                break

    print(f"\nTesting {len(sample_indices)} samples from glosses: {selected_glosses}")
    print(f"Seq len: {seq_len}, Input dim: {input_dim}")

    # Optional: load SMPL-X for mesh rendering
    smpl_x = None
    if render:
        try:
            from generate_smplx_param import load_smplx_model, params_to_mesh, render_smplx_frame, split_params
            from config import WLASL_SMPLX_Config, SignBank_SMPLX_Config
            human_model_path = getattr(dataset, 'cfg', None)
            # Try to get from cfg
            cfg_obj = model.cfg if hasattr(model, 'cfg') else None
            if cfg_obj and hasattr(cfg_obj, 'HUMAN_MODELS_PATH'):
                os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
                smpl_x = load_smplx_model(cfg_obj.HUMAN_MODELS_PATH)
        except Exception as e:
            print(f"Could not load SMPL-X for rendering: {e}")
            smpl_x = None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for idx in sample_indices:
        motion, gloss, length = dataset[idx]
        # motion: (T, D), gloss: str, length: int

        print(f"\n{'#'*60}")
        print(f"# Gloss: {gloss}  (actual length: {length})")
        print(f"{'#'*60}")

        # ---- 1. Ground Truth ----
        gt_np = motion.numpy()
        gt_stats = analyze_temporal_variation(gt_np, gloss, length)
        print_stats("Ground Truth", gt_stats)

        # ---- 2. Reconstruction (encode real motion → mean → decode) ----
        motion_in = motion.unsqueeze(0).to(device)          # (1, T, D)
        pad = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        if length < seq_len:
            pad[0, length:] = True

        recon = model.reconstruct(motion_in, [gloss], pad)  # (1, T, D)
        recon_np = recon.squeeze(0).cpu().numpy()
        recon_stats = analyze_temporal_variation(recon_np, gloss, length)
        print_stats("Reconstruction (encode→mean→decode)", recon_stats)

        # Reconstruction error
        valid = gt_np[:length]
        recon_valid = recon_np[:length]
        mse = ((valid - recon_valid) ** 2).mean()
        print(f"  Reconstruction MSE: {mse:.6f}")

        # ---- 3. Generation (sample z ~ N(0,I) → decode) ----
        gen = model.generate([gloss], seq_len=seq_len, device=device)  # (1, T, D)
        gen_np = gen.squeeze(0).cpu().numpy()
        gen_stats = analyze_temporal_variation(gen_np, gloss)
        print_stats("Generation (z~N(0,I) → decode)", gen_stats)

        # ---- 4. Generation with different z samples ----
        print(f"\n  Checking z-sensitivity (3 different samples):")
        for trial in range(3):
            gen_trial = model.generate([gloss], seq_len=seq_len, device=device)
            gen_trial_np = gen_trial.squeeze(0).cpu().numpy()
            diff_from_first = np.abs(gen_np - gen_trial_np).mean()
            frame_diff = np.abs(np.diff(gen_trial_np, axis=0)).mean()
            print(f"    Trial {trial+1}: mean_diff_from_trial0={diff_from_first:.6f}, "
                  f"frame_diff={frame_diff:.6f}")

        # ---- 5. Latent space analysis ----
        mu, logvar = model.encode(
            motion_in,
            model.condition_proj(model.encode_text([gloss], device)),
            pad
        )
        mu_np = mu.squeeze(0).cpu().numpy()
        logvar_np = logvar.squeeze(0).cpu().numpy()
        std_np = np.exp(0.5 * logvar_np)
        print(f"\n  Latent stats:")
        print(f"    mu:     mean={mu_np.mean():.4f}, std={mu_np.std():.4f}, "
              f"norm={np.linalg.norm(mu_np):.4f}")
        print(f"    sigma:  mean={std_np.mean():.4f}, std={std_np.std():.4f}")
        print(f"    mu range: [{mu_np.min():.4f}, {mu_np.max():.4f}]")
        print(f"    sigma range: [{std_np.min():.4f}, {std_np.max():.4f}]")

        # How far is posterior from prior N(0,I)?
        kl_per_dim = 0.5 * (mu_np**2 + std_np**2 - 2*np.log(std_np) - 1)
        print(f"    KL(posterior || prior): {kl_per_dim.sum():.2f} "
              f"(per-dim mean={kl_per_dim.mean():.4f})")

        # ---- 6. Optional: render first & last frame ----
        if smpl_x and output_dir:
            try:
                import cv2
                gloss_dir = os.path.join(output_dir, gloss)
                os.makedirs(gloss_dir, exist_ok=True)

                for label, seq_data in [("gt", gt_np), ("recon", recon_np), ("gen", gen_np)]:
                    for frame_idx in [0, min(length - 1, seq_len - 1)]:
                        params = split_params(seq_data[frame_idx])
                        vertices, faces = params_to_mesh(smpl_x, params)
                        if np.isnan(vertices).any():
                            print(f"    Render skip {label}_f{frame_idx}: NaN vertices")
                            continue
                        img = render_smplx_frame(vertices, faces, img_w=512, img_h=512)
                        path = os.path.join(gloss_dir, f"{label}_frame{frame_idx:03d}.png")
                        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        print(f"    Saved: {path}")
            except Exception as e:
                print(f"    Render error: {e}")

    # ---- Summary ----
    print(f"\n\n{'='*60}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*60}")
    print("""
If GT has variation but Reconstruction does NOT:
  → Decoder cannot utilize positional encoding.
  → Architecture issue: memory is constant across frames.
  → Fix: add temporal tokens or autoregressive decoding.

If Reconstruction has variation but Generation does NOT:
  → Latent space too spread out, z from N(0,I) is OOD.
  → Fix: increase KL_WEIGHT further, or use KL annealing.

If both Reconstruction and Generation have variation:
  → Model works, but all glosses may map to similar motion.
  → Fix: check condition (CLIP) differentiation.

If z-sensitivity trials all produce identical output:
  → Decoder ignores z entirely (posterior collapse).
  → Fix: KL annealing, or increase latent dim.
""")


def main():
    parser = argparse.ArgumentParser(description="Diagnose reconstruction vs generation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="WLASL_SMPLX",
                        choices=["SignBank_SMPLX", "WLASL_SMPLX"])
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./diagnosis_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset_name == "SignBank_SMPLX":
        cfg = SignBank_SMPLX_Config()
    else:
        cfg = WLASL_SMPLX_Config()

    model, cfg = load_model(cfg, args.checkpoint, device)
    dataset = load_dataset(cfg, args.dataset_name, mode=args.mode)

    run_diagnosis(
        model, dataset, device,
        num_samples=args.num_samples,
        render=args.render_mesh,
        output_dir=args.output_dir if args.render_mesh else None,
    )


if __name__ == "__main__":
    main()
