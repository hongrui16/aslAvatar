"""
ASL Avatar Training Script (V1)

Usage:
    python train.py
    python train.py --debug
    python train.py --batch_size 16 --epochs 100
"""

import os
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger

from dataloader.ASLLVDDataset import ASLLVDSkeletonDataset
from dataloader.SignBankSMPLXDataset import SignBankSMPLXDataset
from aslAvatarModel import ASLAvatarModel
from utils.utils import plot_training_curves, backup_code, collate_fn, create_padding_mask


from conflg import SignBank_SMPLX_Config, ASLLVD_Skeleton3D_Config



class Trainer:
    """Training worker with Accelerate support"""
    
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        if self.dataset_name == "ASLLVD_Skeleton3D":
            Config = ASLLVD_Skeleton3D_Config
        elif self.dataset_name == "SignBank_SMPLX":
            Config = SignBank_SMPLX_Config
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        self.cfg = Config()
        self.cfg.DATASET_NAME = self.dataset_name
        
        self.debug = args.debug
        
        # Override config from args
        if args.batch_size:
            self.cfg.TRAIN_BSZ = args.batch_size
        if args.epochs:
            self.cfg.MAX_EPOCHS = args.epochs
        if args.lr:
            self.cfg.LEARNING_RATE = args.lr
            
        
        # Setup directories
        """Create output directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Add SLURM job ID if available
        slurm_id = os.getenv('SLURM_JOB_ID')
        if slurm_id:
            timestamp += f"_job{slurm_id}"
        
        if self.args.debug:
            timestamp = "debug_" + timestamp
        
        self.logging_dir = os.path.join(
            self.cfg.LOG_DIR,
            self.cfg.PROJECT_NAME,
            self.cfg.DATASET_NAME,
            timestamp
        )
        
        if self.debug:
            self.ckpt_dir = self.logging_dir
        else:
            self.ckpt_dir = os.path.join(
                self.cfg.CKPT_DIR,
                self.cfg.PROJECT_NAME,
                self.cfg.DATASET_NAME,
                timestamp
            )
        
        # Initialize Accelerator
        acc_config = ProjectConfiguration(
            project_dir=self.logging_dir,
            logging_dir=self.logging_dir
        )
        
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.GRAD_ACCUM,
            mixed_precision=self.cfg.MIXED_PRECISION,
            project_config=acc_config
        )
        self.device = self.accelerator.device
        
        
        # Create directories``
        if self.accelerator.is_main_process:        
            os.makedirs(self.logging_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
        
        
        # Setup logging
        self._setup_logging()
        
        # Build components
        self._build_components()
        
        ## backup python files
        if self.accelerator.is_main_process:
            # src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
            src_dir = os.path.dirname(os.path.abspath(__file__))
            dst_dir = os.path.join(self.logging_dir, 'code_backup')
            os.makedirs(dst_dir, exist_ok=True)
            backup_code(
                project_root=src_dir,
                backup_dir=dst_dir,
                logger=self.logger)
            
        
        self.global_step = 0
        self.best_loss = float('inf')
        self.start_epoch = 0

        # Resume from checkpoint if specified
        if args.resume:
            self.load_checkpoint(args.resume, finetune=args.finetune)
            self.cfg.RESUME = args.resume
            self.cfg.FINETUNE = args.finetune

                ## print config to log
        self.logger.info("Config:")
        for k, v in vars(self.cfg).items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("-------------------------------\n")

    def _setup_logging(self):
        """Configure logging"""
        log_file = os.path.join(self.logging_dir, "train.log")
        
        handlers = []
        if self.accelerator.is_main_process:
            handlers = [
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=handlers
        )
        self.logger = get_logger(__name__)
        
        if self.accelerator.is_main_process:
            self.logger.info(f"Logging directory: {self.logging_dir}")
            self.logger.info(f"Checkpoint directory: {self.ckpt_dir}")

    
        

    def _build_components(self):
        """Initialize model, optimizer, and dataloaders"""
        self.logger.info("Building components...")
        
        # Datasets
        self.logger.info("Loading datasets...")
        if self.cfg.DATASET_NAME == "ASLLVD_Skeleton3D":
            train_dataset = ASLLVDSkeletonDataset(mode='train', cfg=self.cfg)
            test_dataset = ASLLVDSkeletonDataset(mode='test', cfg=self.cfg)
        elif self.cfg.DATASET_NAME == "SignBank_SMPLX":
            train_dataset = SignBankSMPLXDataset(mode='train', cfg=self.cfg)
            test_dataset = None  # TODO: implement test dataset for SignBank_SMPLX
        else:
            raise ValueError(f"Unknown dataset: {self.cfg.DATASET_NAME}")
            
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.TRAIN_BSZ,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        self.logger.info(f"Train samples: {len(train_dataset)}, Train batches: {len(self.train_loader)}")
        

        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.cfg.EVAL_BSZ,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.cfg.NUM_WORKERS,
                pin_memory=True
            )
            self.logger.info(f"Test samples: {len(test_dataset)}, Test batches: {len(self.test_loader)}")
        else:
            self.test_loader = None
        
                
        
        # Model
        self.logger.info("Building model...")
        self.model = ASLAvatarModel(self.cfg)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Optimizer (only trainable params, excluding frozen CLIP)
        trainable_params_list = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params_list,
            lr=self.cfg.LEARNING_RATE,
            weight_decay=self.cfg.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        num_training_steps = self.cfg.MAX_EPOCHS * len(self.train_loader)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=self.cfg.LEARNING_RATE * 0.01
        )
        
        # Prepare with Accelerator
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.lr_scheduler
        )
        
        if self.test_loader is not None:
            self.test_loader = self.accelerator.prepare(self.test_loader)

    

    def save_checkpoint(self, epoch, metrics=None, is_best=False):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save only trainable parameters (exclude frozen CLIP)
        state_dict = {
            k: v for k, v in unwrapped_model.state_dict().items()
            if "text_encoder" not in k and "tokenizer" not in k
        }
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': vars(self.cfg),
            'metrics': metrics,
            'best_loss': self.best_loss
            
        }
        
        # Save regular checkpoint
        ckpt_path = os.path.join(self.ckpt_dir, f"newest_model.pt")
        torch.save(checkpoint, ckpt_path)
        self.logger.info(f"Saved checkpoint: {ckpt_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.ckpt_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path, finetune=False):
        """
        Load checkpoint to resume training or finetune.
        
        Args:
            checkpoint_path: Path to checkpoint file
            finetune: If True, reset training state (epoch, optimizer, etc.)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint to CPU first to save GPU memory
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Load model state (partial, excluding CLIP)
        model_state = ckpt['model_state_dict']
        current_state = unwrapped_model.state_dict()
        
        # Update with saved weights (only matching keys)
        loaded_keys = []
        skipped_keys = []
        for key in model_state:
            if key in current_state:
                if current_state[key].shape == model_state[key].shape:
                    current_state[key] = model_state[key]
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(key)
                    self.logger.warning(f"Shape mismatch for {key}: "
                                       f"{current_state[key].shape} vs {model_state[key].shape}")
        
        unwrapped_model.load_state_dict(current_state, strict=False)
        self.logger.info(f"Loaded model weights (loaded={len(loaded_keys)}, skipped={len(skipped_keys)})")
        
        if finetune:
            # Finetune mode: reset training state, only keep model weights
            self.logger.info("Finetune mode: resetting training state")
            self.global_step = 0
            self.start_epoch = 0
            self.best_loss = float('inf')
        else:
            # Resume mode: restore full training state
            if 'optimizer_state_dict' in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    self.logger.info("Loaded optimizer state")
                except Exception as e:
                    self.logger.warning(f"Could not load optimizer state: {e}")
            
            if 'scheduler_state_dict' in ckpt:
                try:
                    self.lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    self.logger.info("Loaded scheduler state")
                except Exception as e:
                    self.logger.warning(f"Could not load scheduler state: {e}")
            
            self.start_epoch = ckpt.get('epoch', -1) + 1
            self.global_step = ckpt.get('global_step', 0)
            self.best_loss = ckpt.get('best_loss', float('inf'))
            self.logger.info(f"Resumed from epoch {self.start_epoch}, "
                           f"global_step {self.global_step}, best_loss {self.best_loss:.4f}")
        
        # Free memory
        del ckpt
        torch.cuda.empty_cache()
            
    def get_mask_ratio(self, epoch):
        mask_increment = self.cfg.MASK_INCREMENT
        mask_step_epochs = self.cfg.MASK_STEP_EPOCHS
        max_mask_ratio = self.cfg.MAX_MASK_RATIO
        return min(mask_increment * (epoch // mask_step_epochs), max_mask_ratio)


    def apply_masking(self, motion, ratio, padding_mask=None):
        """
        Apply random frame masking for curriculum learning.
        
        Args:
            motion: (B, T, D) - pose sequences
            ratio: float - fraction of frames to mask
            padding_mask: (B, T) - True where padded (don't mask padding)
        
        Returns:
            masked_motion: (B, T, D)
        """
        if ratio <= 0:
            return motion
        
        B, T, D = motion.shape
        device = motion.device
        
        # Random mask: True = keep, False = mask
        keep_mask = torch.rand(B, T, device=device) >= ratio
        
        # Don't mask already padded positions
        if padding_mask is not None:
            keep_mask = keep_mask | padding_mask  # Keep if padding or randomly kept
        
        # Apply mask
        masked_motion = motion * keep_mask.unsqueeze(-1).float()
        
        return masked_motion

    def compute_loss(self, recon, target, mu, logvar, padding_mask):
        """
        Compute CVAE loss = reconstruction + KL divergence.
        
        Args:
            recon: (B, T, D) - reconstructed motion
            target: (B, T, D) - ground truth motion
            mu: (B, latent_dim) - mean
            logvar: (B, latent_dim) - log variance
            padding_mask: (B, T) - True where padded
        
        Returns:
            total_loss, mse_loss, kl_loss
        """
        # Reconstruction loss (MSE, excluding padding)
        mse = F.mse_loss(recon, target, reduction='none')  # (B, T, D)
        
        # Create mask for valid (non-padded) positions
        valid_mask = ~padding_mask  # (B, T), True where valid
        valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(mse)  # (B, T, D)
        
        # Masked MSE
        mse_masked = mse * valid_mask_expanded.float()
        num_valid = valid_mask_expanded.sum()
        mse_loss = mse_masked.sum() / (num_valid + 1e-8)
        
        # KL divergence: KL(N(mu, sigma) || N(0, I))
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)  # Normalize by batch size
        
        # Total loss
        total_loss = mse_loss + self.cfg.KL_WEIGHT * kl_loss
        
        return total_loss, mse_loss, kl_loss

    def train_epoch(self, epoch):
        """Run one training epoch"""
        self.model.train()
        
        mask_ratio = self.get_mask_ratio(epoch)
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_kl = 0.0
        num_batches = 0
        
        num_prints = 50
        total_steps = len(self.train_loader)
        print_every = max(total_steps // num_prints, 1)

        progress_bar = tqdm(
            total=num_prints,
            disable=not self.accelerator.is_local_main_process,
            desc=f"Epoch {epoch}"
        )
        
        for step, batch in enumerate(self.train_loader):
            if step % print_every == 0 and progress_bar.n < num_prints:
                progress_bar.update(1)
            with self.accelerator.accumulate(self.model):
                motion, labels, lengths = batch
                B, T, _ = motion.shape
                
                # Create padding mask
                padding_mask = create_padding_mask(lengths, T, self.device)
                
                # Apply curriculum masking to encoder input
                input_motion = self.apply_masking(motion, mask_ratio, padding_mask)
                
                # Forward pass
                recon_motion, mu, logvar = self.model(input_motion, labels, padding_mask)
                
                # Compute loss (compare with original unmasked motion)
                loss, mse, kl = self.compute_loss(recon_motion, motion, mu, logvar, padding_mask)
                
                # Backward
                self.accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 

                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # Logging
                epoch_loss += loss.item()
                epoch_mse += mse.item()
                epoch_kl += kl.item()
                num_batches += 1
                
                    
                if step == 0:
                    mem = torch.cuda.max_memory_allocated(self.device) / 2**30
                    self.logger.info(f"[epoch {epoch}] Peak GPU: {mem:.2f} GB")

                if self.debug and step >= 10:
                    break
                
        progress_bar.close()
        # Return epoch averages
        return {
            'loss': epoch_loss / num_batches,
            'mse': epoch_mse / num_batches,
            'kl': epoch_kl / num_batches,
            'mask_ratio': mask_ratio
        }



    def train(self):
        """Main training loop"""
        self.logger.info("=" * 60)
        self.logger.info("Starting training")
        self.logger.info(f"  Epochs: {self.cfg.MAX_EPOCHS}")
        self.logger.info(f"  Batch size: {self.cfg.TRAIN_BSZ}")
        self.logger.info(f"  Learning rate: {self.cfg.LEARNING_RATE}")
        self.logger.info(f"  KL weight: {self.cfg.KL_WEIGHT}")
        self.logger.info(f"  Curriculum learning: {self.cfg.USE_CURRICULUM}")
        self.logger.info("=" * 60)
        

        train_hist = {'total': [],
                'rec': [],
                'kl': [],
                'mask_ratio': []}      

        eval_hist = {
            'total': [],
            'rec': [],
            'kl': []
        }
        
        for epoch in range(self.start_epoch, self.cfg.MAX_EPOCHS):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            train_hist['total'].append(train_metrics['loss'])
            train_hist['rec'].append(train_metrics['mse'])
            train_hist['kl'].append(train_metrics['kl'])
            train_hist['mask_ratio'].append(train_metrics['mask_ratio'])
            
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Train Epoch {epoch+1}/{self.cfg.MAX_EPOCHS}: loss={train_metrics['loss']:.4f}, "
                    f"rec={train_metrics['mse']:.4f}, kl={train_metrics['kl']:.4f}, mask_ratio={train_metrics['mask_ratio']:.2f}"
                )
            

            if not self.test_loader is None:
                # Evaluate
                eval_metrics = self.evaluate(epoch)
                
                eval_hist['total'].append(eval_metrics['loss'])
                eval_hist['rec'].append(eval_metrics['mse'])
                eval_hist['kl'].append(eval_metrics['kl'])
            else:
                eval_metrics = train_metrics
                eval_hist = None
                
            # Check if best
            is_best = eval_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = eval_metrics['loss']

            self.save_checkpoint(epoch, eval_metrics, is_best)

        
        
            if self.accelerator.is_main_process and train_hist['total']:
                fig = os.path.join(self.logging_dir, 'training_curves.png')
                if os.path.exists(fig):
                    os.remove(fig)
                plot_training_curves(fig, self.start_epoch,
                                    train_hist, eval_hist)
                self.logger.info(f"Saved curves: {fig}")
            
            if self.debug and epoch >= 10:
                break
        self.logger.info("Training complete!")

        
        self.accelerator.end_training()
        self.logger.info("Training completed!")


    
    @torch.no_grad()
    def evaluate(self, epoch):
        """Run evaluation on test set"""
        self.model.eval()
        
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        num_batches = 0
        
        for batch in tqdm(self.test_loader, desc="Evaluating", 
                          disable=not self.accelerator.is_local_main_process):
            motion, labels, lengths = batch
            B, T, _ = motion.shape
            
            padding_mask = create_padding_mask(lengths, T, self.device)
            
            # No masking during evaluation
            recon_motion, mu, logvar = self.model(motion, labels, padding_mask)
            loss, mse, kl = self.compute_loss(recon_motion, motion, mu, logvar, padding_mask)
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kl += kl.item()
            num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'kl': total_kl / num_batches
        }
        
        if self.accelerator.is_main_process:
            self.accelerator.log({
                "eval/loss": metrics['loss'],
                "eval/mse": metrics['mse'],
                "eval/kl": metrics['kl']
            }, step=self.global_step)
            
            self.logger.info(
                f"Eval Epoch {epoch+1}: loss={metrics['loss']:.4f}, "
                f"mse={metrics['mse']:.4f}, kl={metrics['kl']:.4f}"
            )
        
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train ASL Avatar Model")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--finetune", action="store_true", help="Finetune mode: load weights but reset training state")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--dataset", type=str, default="SignBank_SMPLX", choices=["ASLLVD_Skeleton3D", "SignBank_SMPLX"], help="Dataset to use")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()