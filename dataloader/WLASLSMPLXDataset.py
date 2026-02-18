import os
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class WLASLSMPLXDataset(Dataset):
    """
    Dataset for WLASL SMPL-X parameters.

    File structure:
        smplx_params/
            book/
                06297/
                    book_06297_000000_p0.npz
                    book_06297_000012_p0.npz
                    ...
                06301/
                    ...
            drink/
                07102/
                    ...

    One sample = one video (gloss/video_id/*.npz sorted by frame index).
    Returns: (pose_tensor, label, length)
        - pose_tensor: (target_seq_len, input_dim)
        - label: int (gloss index)
        - length: int (actual frames before padding)

    Each .npz contains SMPL-X parameters for a single frame:
        - smplx_root_pose:   (3,)     Global orientation (axis-angle)
        - smplx_body_pose:   (21, 3)  Body joint rotations (axis-angle)
        - smplx_lhand_pose:  (15, 3)  Left hand joint rotations
        - smplx_rhand_pose:  (15, 3)  Right hand joint rotations
        - smplx_jaw_pose:    (3,)     Jaw rotation
        - smplx_shape:       (10,)    Body shape (betas)
        - smplx_expr:        (10,)    Facial expression coefficients
        - cam_trans:         (3,)     Camera translation
        - focal:             (2,)     Focal length [fx, fy]
        - princpt:           (2,)     Principal point [cx, cy]
        - smplx_joint_cam,   (1, 137, 3)  3D joints in camera space
    
    One "sample" = all keyframes of a single gloss, sorted by frame index,
    yielding a temporal sequence of SMPL-X poses.
    
    Pose Feature Vector (per frame):
        root_pose (3) + body_pose (63) + lhand_pose (45) + rhand_pose (45) + jaw_pose (3) = 159
        Optional: + expression (10) + shape (10) + cam_trans (3) = 182
    """

    # ==================== Pose Dimensions ====================
    ROOT_DIM = 3           # 1 joint × 3 (axis-angle)
    BODY_DIM = 63          # 21 joints × 3
    LHAND_DIM = 45         # 15 joints × 3
    RHAND_DIM = 45         # 15 joints × 3
    JAW_DIM = 3            # 1 joint × 3
    POSE_DIM = ROOT_DIM + BODY_DIM + LHAND_DIM + RHAND_DIM + JAW_DIM  # 159

    EXPR_DIM = 10
    SHAPE_DIM = 10
    CAM_TRANS_DIM = 3

    # Joint counts (for reshaping)
    BODY_JOINTS = 21
    LHAND_JOINTS = 15
    RHAND_JOINTS = 15

    def __init__(self, mode='train', cfg=None, logger = None):
        """
        Args:
            mode: 'train' or 'test'
            cfg: Config object with at least:
                - SMPLX_DIR: path to smplx_params root directory
                - TRAIN_SPLIT_FILE / TEST_SPLIT_FILE: split file paths
                - MAX_SEQ_LEN: max sequence length
                - MIN_SEQ_LEN: min sequence length (for interpolation)
                - INTERPOLATE_SHORT_SEQ: bool
                - INCLUDE_EXPRESSION: bool (optional, default True)
                - INCLUDE_SHAPE: bool (optional, default False)
                - INCLUDE_CAM_TRANS: bool (optional, default False)
        """
        assert mode in ['train', 'test'], f"mode must be 'train' or 'test', got {mode}"
        # self.cfg = cfg
        self.mode = mode
        self.logger = logger

        # Feature selection
        self.include_expr = getattr(cfg, 'INCLUDE_EXPRESSION', False)
        self.include_shape = getattr(cfg, 'INCLUDE_SHAPE', False)
        self.include_cam_trans = getattr(cfg, 'INCLUDE_CAM_TRANS', False)

        # Compute actual input dimension based on config
        self.input_dim = self.POSE_DIM
        if self.include_expr:
            self.input_dim += self.EXPR_DIM
        if self.include_shape:
            self.input_dim += self.SHAPE_DIM
        if self.include_cam_trans:
            self.input_dim += self.CAM_TRANS_DIM
        
        self.n_joints = 53
        self.n_feats = 3
        self.target_seq_len = getattr(cfg, 'TARGET_SEQ_LEN', 48) if cfg is not None else 48 
        self.min_frames = 5
        self.root_dir = getattr(cfg, 'ROOT_DIR', './smplx_params') if cfg is not None else './smplx_params'

        self.smplx_params_dir = os.path.join(self.root_dir, mode, 'smplx_params')
        self.data_list = []      # [(gloss, video_dir_path), ...]
        self.gloss_to_idx = {}   # {gloss_name: int}

        self.gloss_name_list = []
        
        self._check_dirs()
        self._load_all_samples()

    # ==================== Initialization ====================

    def _check_dirs(self):
        """Verify data directory exists."""
        if not os.path.exists(self.smplx_params_dir):
            raise FileNotFoundError(f"SMPL-X directory not found: {self.smplx_params_dir}")
        
    def _load_all_samples(self):
        if not os.path.exists(self.smplx_params_dir):
            raise FileNotFoundError(f"Not found: {self.smplx_params_dir}")

        skipped = 0
        for gloss in sorted(os.listdir(self.smplx_params_dir)):
            gloss_path = os.path.join(self.smplx_params_dir, gloss)
            if not os.path.isdir(gloss_path):
                continue

            for video_id in sorted(os.listdir(gloss_path)):
                video_path = os.path.join(gloss_path, video_id)
                if not os.path.isdir(video_path):
                    continue

                n = sum(1 for f in os.listdir(video_path) if f.endswith('.npz'))
                if n < self.min_frames:
                    skipped += 1
                    continue

                if gloss not in self.gloss_to_idx:
                    self.gloss_to_idx[gloss] = len(self.gloss_to_idx)
                self.data_list.append((gloss, video_path))
        self.gloss_name_list = [g.lower() for g, _ in sorted(self.gloss_to_idx.items(), key=lambda x: x[1])]
        
        
        self.logger.info(f"[{self.mode}] {len(self.data_list)} videos, "
              f"{len(self.gloss_to_idx)} glosses "
              f"(skipped {skipped} < {self.min_frames} frames)")
        self.logger.info(f"[{self.mode}] Glosses: {sorted(self.gloss_to_idx.keys())}")
        
        
    # ==================== Data Loading ====================
    def _get_sorted_npz(self, video_dir):
            """Return sorted (frame_idx, filepath) list."""
            frames = []
            for fname in os.listdir(video_dir):
                if not fname.endswith('.npz'):
                    continue
                # 格式: gloss_videoID_frameID_p0.npz
                parts = fname.replace('.npz', '').split('_')
                frame_idx = int(parts[2])
                frames.append((frame_idx, os.path.join(video_dir, fname)))
            frames.sort(key=lambda x: x[0])
            return frames

    def _load_frame(self, npz_path):
        """
        Load SMPL-X parameters from a single .npz file.
        
        Returns:
            torch.Tensor: (input_dim,) — concatenated pose features for one frame
        """
        data = np.load(npz_path, allow_pickle=True)

        # Core pose parameters (always included)
        root_pose = data.get('smplx_root_pose', np.zeros(3))                # (3,)
        body_pose = data.get('smplx_body_pose', np.zeros((21, 3)))           # (21, 3)
        lhand_pose = data.get('smplx_lhand_pose', np.zeros((15, 3)))         # (15, 3)
        rhand_pose = data.get('smplx_rhand_pose', np.zeros((15, 3)))         # (15, 3)
        jaw_pose = data.get('smplx_jaw_pose', np.zeros(3))                   # (3,)

        # Flatten all to 1D
        features = [
            root_pose.flatten(),      # (3,)
            body_pose.flatten(),      # (63,)
            lhand_pose.flatten(),     # (45,)
            rhand_pose.flatten(),     # (45,)
            jaw_pose.flatten(),       # (3,)
        ]

        # Optional features
        if self.include_expr:
            expr = data.get('smplx_expr', np.zeros(10))
            features.append(expr.flatten()[:self.EXPR_DIM])  # (10,)

        if self.include_shape:
            shape = data.get('smplx_shape', np.zeros(10))
            features.append(shape.flatten()[:self.SHAPE_DIM])  # (10,)

        if self.include_cam_trans:
            cam_trans = data.get('cam_trans', np.zeros(3))
            features.append(cam_trans.flatten()[:self.CAM_TRANS_DIM])  # (3,)

        feature_vec = np.concatenate(features, axis=0).astype(np.float32)
        return torch.tensor(feature_vec, dtype=torch.float32)

    # ==================== Normalization ====================

    def _normalize_pose(self, pose_seq):
        """
        Normalize SMPL-X pose sequence.
        
        For axis-angle rotations, we normalize the root orientation
        relative to the first frame (root-relative representation),
        making the sequence translation/rotation invariant.
        
        Args:
            pose_seq: (T, D) — pose sequence
            
        Returns:
            normalized_seq: (T, D)
        """
        T, D = pose_seq.shape

        # Extract root orientation (first 3 dims = axis-angle)
        root_poses = pose_seq[:, :self.ROOT_DIM].clone()  # (T, 3)

        # Make root orientation relative to first frame
        # In axis-angle: subtract first frame's rotation (approximate for small rotations)
        # For more accurate version, convert to rotation matrices
        first_root = root_poses[0:1, :]  # (1, 3)
        pose_seq = pose_seq.clone()
        pose_seq[:, :self.ROOT_DIM] = root_poses - first_root

        # If cam_trans is included, also make it relative to first frame
        if self.include_cam_trans:
            cam_start = self.POSE_DIM
            if self.include_expr:
                cam_start += self.EXPR_DIM
            if self.include_shape:
                cam_start += self.SHAPE_DIM
            cam_end = cam_start + self.CAM_TRANS_DIM
            cam_trans = pose_seq[:, cam_start:cam_end].clone()
            first_cam = cam_trans[0:1, :]
            pose_seq[:, cam_start:cam_end] = cam_trans - first_cam

        return pose_seq

        # ==================== Main Interface ====================

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        gloss, video_dir = self.data_list[idx]


        frames = self._get_sorted_npz(video_dir)
        if not frames:
            return torch.zeros(self.target_seq_len, self.input_dim), gloss, 1

        poses = []
        for _, path in frames:
            try:
                poses.append(self._load_frame(path))
            except Exception as e:
                print(f"Warning: {path}: {e}")

        if not poses:
            return torch.zeros(self.target_seq_len, self.input_dim), gloss, 1

        seq = self._normalize_pose(torch.stack(poses))
        actual_len = seq.shape[0]

        if actual_len > self.target_seq_len:
            idx_sub = torch.linspace(0, actual_len - 1, self.target_seq_len).long()
            seq = seq[idx_sub]
            actual_len = self.target_seq_len
        elif actual_len < self.target_seq_len:
            pad = torch.zeros(self.target_seq_len - actual_len, self.input_dim)
            seq = torch.cat([seq, pad], dim=0)

        return seq, gloss, actual_len

    def get_num_classes(self):
        return len(self.gloss_to_idx)

    def get_gloss_name(self, idx):
        for g, i in self.gloss_to_idx.items():
            if i == idx:
                return g
        return None

    def get_gloss_samples(self, gloss_name):
        return [i for i, (g, _) in enumerate(self.data_list) if g == gloss_name]

    def get_feature_indices(self):
        """
        Return a dict mapping parameter names to their index ranges
        in the feature vector. Useful for extracting specific params later.
        """
        idx = 0
        indices = {}

        indices['root_pose'] = (idx, idx + self.ROOT_DIM)
        idx += self.ROOT_DIM

        indices['body_pose'] = (idx, idx + self.BODY_DIM)
        idx += self.BODY_DIM

        indices['lhand_pose'] = (idx, idx + self.LHAND_DIM)
        idx += self.LHAND_DIM

        indices['rhand_pose'] = (idx, idx + self.RHAND_DIM)
        idx += self.RHAND_DIM

        indices['jaw_pose'] = (idx, idx + self.JAW_DIM)
        idx += self.JAW_DIM

        if self.include_expr:
            indices['expression'] = (idx, idx + self.EXPR_DIM)
            idx += self.EXPR_DIM

        if self.include_shape:
            indices['shape'] = (idx, idx + self.SHAPE_DIM)
            idx += self.SHAPE_DIM

        if self.include_cam_trans:
            indices['cam_trans'] = (idx, idx + self.CAM_TRANS_DIM)
            idx += self.CAM_TRANS_DIM

        return indices

    def to_smplx_dict(self, feature_vec):
        """
        Convert a feature vector back to a dict of SMPL-X parameters.
        Useful for visualization / mesh reconstruction.
        
        Args:
            feature_vec: (D,) or (T, D) tensor
            
        Returns:
            dict of parameter tensors
        """
        if feature_vec.dim() == 1:
            feature_vec = feature_vec.unsqueeze(0)  # (1, D)

        indices = self.get_feature_indices()
        result = {}

        s, e = indices['root_pose']
        result['smplx_root_pose'] = feature_vec[:, s:e]  # (T, 3)

        s, e = indices['body_pose']
        result['smplx_body_pose'] = feature_vec[:, s:e].view(-1, self.BODY_JOINTS, 3)  # (T, 21, 3)

        s, e = indices['lhand_pose']
        result['smplx_lhand_pose'] = feature_vec[:, s:e].view(-1, self.LHAND_JOINTS, 3)  # (T, 15, 3)

        s, e = indices['rhand_pose']
        result['smplx_rhand_pose'] = feature_vec[:, s:e].view(-1, self.RHAND_JOINTS, 3)  # (T, 15, 3)

        s, e = indices['jaw_pose']
        result['smplx_jaw_pose'] = feature_vec[:, s:e]  # (T, 3)

        if 'expression' in indices:
            s, e = indices['expression']
            result['smplx_expr'] = feature_vec[:, s:e]  # (T, 10)

        if 'shape' in indices:
            s, e = indices['shape']
            result['smplx_shape'] = feature_vec[:, s:e]  # (T, 10)

        if 'cam_trans' in indices:
            s, e = indices['cam_trans']
            result['cam_trans'] = feature_vec[:, s:e]  # (T, 3)

        return result

