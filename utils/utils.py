import os
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
# Plot training curves
from torch.nn.utils.rnn import pad_sequence
import torch


def plot_training_curves(fig_path, start_epoch, train_hist, eval_hist = None):    
    epochs = list(range(start_epoch, start_epoch + len(train_hist['total'])))
    
    num_fig = 2
    if eval_hist is not None and len(eval_hist['total']) > 0:
        num_row = 2
    else:
        num_row = 1
    if 'kl' in train_hist:
        num_fig += 1
    if 'mask_ratio' in train_hist:
        num_fig += 1

    
    fig, axes = plt.subplots(num_row, num_fig, figsize=(15, 4))
    # make axes always 1D
    axes = np.array(axes).reshape(-1)


    # print("train_hist['total']:", train_hist['total'])
    # Total loss
    axes[0].plot(epochs, train_hist['total'], 'b-o', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train Total Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Rec loss
    axes[1].plot(epochs, train_hist['rec'], 'g-o', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Train Rec Loss')
    axes[1].grid(True, alpha=0.3)
    fig_idx = 1
    
    if 'kl' in train_hist:
        # KL loss
        axes[2].plot(epochs, train_hist['kl'], 'r-o', linewidth=2, markersize=4)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Train KL Loss')
        axes[2].grid(True, alpha=0.3)
        fig_idx = 2
        
    if 'mask_ratio' in train_hist:        
        idx = fig_idx + 1
        # Mask Ratio
        axes[idx].plot(epochs, train_hist['mask_ratio'], 'm-o', linewidth=2, markersize=4)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Mask Ratio')
        axes[idx].set_title('Train Mask Ratio')
        axes[idx].grid(True, alpha=0.3)
        fig_idx += 1
        
    if eval_hist is not None and len(eval_hist['total']) > 0:
        axes[fig_idx + 1].plot(epochs, eval_hist['total'], 'b-s', linewidth=2, markersize=4)
        axes[fig_idx + 1].set_xlabel('Epoch')
        axes[fig_idx + 1].set_ylabel('Loss')
        axes[fig_idx + 1].set_title('Eval Total Loss')
        axes[fig_idx + 1].grid(True, alpha=0.3) 
        
        fig_idx += 1
        axes[fig_idx + 1].plot(epochs, eval_hist['rec'], 'g-s', linewidth=2, markersize=4)
        axes[fig_idx + 1].set_xlabel('Epoch')
        axes[fig_idx + 1].set_ylabel('Loss')
        axes[fig_idx + 1].set_title('Eval Rec Loss')
        axes[fig_idx + 1].grid(True, alpha=0.3)
        fig_idx += 1
        
        if 'kl' in eval_hist:
            axes[fig_idx + 1].plot(epochs, eval_hist['kl'], 'r-s', linewidth=2, markersize=4)
            axes[fig_idx + 1].set_xlabel('Epoch')
            axes[fig_idx + 1].set_ylabel('Loss')
            axes[fig_idx + 1].set_title('Eval KL Loss')
            axes[fig_idx + 1].grid(True, alpha=0.3)

    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(fig_path, dpi=150)
    plt.close()

        
        


def backup_code(
    project_root,
    backup_dir,
    logger,
    exclude_dirs=('zlog', 'log', 'temp', 'output')
):
    """
    Backup all .py files under project_root to backup_dir,
    skipping specified directories.
    """
    project_root = Path(project_root).resolve()
    dst_root = Path(backup_dir).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    exclude_dirs = set(exclude_dirs)

    for file_path in project_root.rglob('*.py'):
        # 跳过指定目录
        if any(part in exclude_dirs for part in file_path.parts):
            continue

        relative_path = file_path.relative_to(project_root)
        dst_path = dst_root / relative_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open('r', encoding='utf-8', errors='ignore') as src_file:
            with dst_path.open('w', encoding='utf-8') as dst_file:
                dst_file.write(src_file.read())

    logger.info(f'Backed up code to: {dst_root}')




def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of (pose_tensor, label, length) tuples
        
    Returns:
        poses_padded: (B, max_T, D) - padded pose sequences
        labels: List[str] - gloss labels
        lengths: (B,) - actual lengths
    """
    poses, labels, lengths = zip(*batch)
    
    # Pad sequences to max length in batch
    poses_padded = pad_sequence(poses, batch_first=True, padding_value=0.0)
    
    # Convert lengths to tensor
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return poses_padded, list(labels), lengths_tensor


def create_padding_mask(lengths, max_len, device):
    """
    Create padding mask from lengths.
    
    Args:
        lengths: (B,) - actual sequence lengths
        max_len: int - padded sequence length
        device: torch device
        
    Returns:
        mask: (B, max_len) - True where padded
    """
    B = lengths.shape[0]
    indices = torch.arange(max_len, device=device).expand(B, -1)
    mask = indices >= lengths.unsqueeze(1)
    return mask


