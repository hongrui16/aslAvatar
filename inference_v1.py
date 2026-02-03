"""
Inference script for ASL Avatar Model

Usage:
    python inference.py --checkpoint path/to/checkpoint.pt --gloss "hello"
    python inference.py --checkpoint path/to/checkpoint.pt --gloss "hello,thank you,please" --num_samples 3
"""

import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path

from aslAvatarModel import ASLAvatarModel


class Config:
    """Must match training config"""
    def __init__(self):
        self.INPUT_DIM = 390
        self.MAX_SEQ_LEN = 200
        self.CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
        self.CLIP_DIM = 512
        self.LATENT_DIM = 256
        self.MODEL_DIM = 512
        self.N_HEADS = 8
        self.N_LAYERS = 4
        self.DROPOUT = 0.1


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    cfg = Config()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Override config if saved in checkpoint
    if 'config' in checkpoint:
        saved_cfg = checkpoint['config']
        for key, value in saved_cfg.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    
    # Build model
    model = ASLAvatarModel(cfg)
    
    # Load weights (partial, excluding CLIP)
    model_state = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle missing keys (CLIP weights are loaded from pretrained)
    current_state = model.state_dict()
    for key in model_state:
        if key in current_state:
            current_state[key] = model_state[key]
    
    model.load_state_dict(current_state, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Global step: {checkpoint.get('global_step', 'unknown')}")
    
    return model, cfg


def generate(model, glosses, seq_len=100, num_samples=1, device='cuda'):
    """
    Generate motion sequences for given glosses.
    
    Args:
        model: trained ASLAvatarModel
        glosses: list of gloss strings
        seq_len: output sequence length
        num_samples: number of samples per gloss
        device: torch device
    
    Returns:
        dict: {gloss: list of (num_samples, seq_len, 390) arrays}
    """
    results = {}
    
    for gloss in glosses:
        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                motion = model.generate([gloss], seq_len=seq_len, device=device)
                samples.append(motion.cpu().numpy()[0])  # (seq_len, 390)
        
        results[gloss] = samples
        print(f"Generated {num_samples} sample(s) for '{gloss}'")
    
    return results


def save_motion(motion, output_path, gloss=None):
    """
    Save generated motion to JSON format compatible with visualization.
    
    Args:
        motion: (T, 390) numpy array
        output_path: output file path
        gloss: optional gloss label
    """
    T = motion.shape[0]
    
    # Reshape to (T, 130, 3)
    motion_reshaped = motion.reshape(T, -1, 3)
    
    # Split into body parts
    body_joints = 18
    face_joints = 70
    hand_joints = 21
    
    frames = []
    idx = 0
    for t in range(T):
        frame_data = {
            'frame_index': t,
            'skeleton': {
                'body': {
                    'x': motion_reshaped[t, :body_joints, 0].tolist(),
                    'y': motion_reshaped[t, :body_joints, 1].tolist(),
                    'z': motion_reshaped[t, :body_joints, 2].tolist()
                },
                'face': {
                    'x': motion_reshaped[t, body_joints:body_joints+face_joints, 0].tolist(),
                    'y': motion_reshaped[t, body_joints:body_joints+face_joints, 1].tolist(),
                    'z': motion_reshaped[t, body_joints:body_joints+face_joints, 2].tolist()
                },
                'hand_left': {
                    'x': motion_reshaped[t, body_joints+face_joints:body_joints+face_joints+hand_joints, 0].tolist(),
                    'y': motion_reshaped[t, body_joints+face_joints:body_joints+face_joints+hand_joints, 1].tolist(),
                    'z': motion_reshaped[t, body_joints+face_joints:body_joints+face_joints+hand_joints, 2].tolist()
                },
                'hand_right': {
                    'x': motion_reshaped[t, -hand_joints:, 0].tolist(),
                    'y': motion_reshaped[t, -hand_joints:, 1].tolist(),
                    'z': motion_reshaped[t, -hand_joints:, 2].tolist()
                }
            }
        }
        frames.append(frame_data)
    
    output = {
        'gloss': gloss or 'generated',
        'num_frames': T,
        'frames': frames
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved motion to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ASL motions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--gloss", type=str, required=True, help="Comma-separated list of glosses")
    parser.add_argument("--seq_len", type=int, default=100, help="Output sequence length")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per gloss")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model, cfg = load_model(args.checkpoint, args.device)
    
    # Parse glosses
    glosses = [g.strip() for g in args.gloss.split(',')]
    print(f"Generating for glosses: {glosses}")
    
    # Generate
    results = generate(model, glosses, args.seq_len, args.num_samples, args.device)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    for gloss, samples in results.items():
        for i, motion in enumerate(samples):
            filename = f"{gloss}_sample{i+1}.json"
            output_path = os.path.join(args.output_dir, filename)
            save_motion(motion, output_path, gloss)
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()