import os
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset




class ASLLVDSkeletonDataset(Dataset):
    """
    Dataset for ASL-Skeleton3D and ASL-Phono data (ASLLVD).
    
    Expected file structure:
        ASL-Skeleton3D/
            gloss1-123456.json
            gloss1-123457.json
            gloss2-234567.json
            ...
        ASL-Phono/
            gloss1-123456.json  (matching filenames)
            ...
    
    Each gloss has multiple samples. For train/test split:
    - 1 sample per gloss goes to test
    - Remaining samples go to train
    
    Note: Original data has very few frames (2-4 per sample at fps=3).
    We interpolate to increase sequence length for better training.
    
    Joint Selection:
    - Body: upper body only (14 joints)
    - Face: key landmarks only (16 joints) 
    - Hands: full (21 joints each)
    Total: 14 + 16 + 21 + 21 = 72 joints = 216 dims
    """
    
    # ==================== Joint Configuration ====================
    # Body: upper body joints only
    # Original: 0:nose, 1:neck, 2:shoulder_r, 3:elbow_r, 4:wrist_r, 5:shoulder_l, 6:elbow_l, 7:wrist_l
    #           8:hip_r, 9:knee_r, 10:ankle_r, 11:hip_l, 12:knee_l, 13:ankle_l, 14:eye_r, 15:eye_l, 16:ear_r, 17:ear_l
    BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 15, 16, 17]  # 14 joints (include hip for normalization)
    BODY_JOINTS = len(BODY_INDICES)  # 14
    
    # Face: key landmarks only (from 70 original points)
    # chin(3) + nose(3) + eye_corners(4) + lips_outer_key(4) + eye_balls(2) = 16
    FACE_INDICES = [
        7, 8, 9,           # face_chin_1, face_chin_2, face_chin_3
        27, 29, 33,        # nose_top, nose_middle_2, nose_bottom
        36, 39,            # eye_left_corner_right, eye_left_corner_left
        42, 45,            # eye_right_corner_right, eye_right_corner_left
        48, 51, 54, 57,    # lips_outer_right, lips_outer_top, lips_outer_left, lips_outer_bottom
        68, 69             # eye_ball_right, eye_ball_left
    ]
    FACE_JOINTS = len(FACE_INDICES)  # 16
    
    # Hands: full 21 joints each
    HAND_JOINTS = 21
    
    # Total joints
    TOTAL_JOINTS = BODY_JOINTS + FACE_JOINTS + HAND_JOINTS * 2  # 14 + 16 + 42 = 72
    
    # Hip indices in the SELECTED body joints (for normalization)
    # In original: hip_r=8, hip_l=11
    # In our selection [0,1,2,3,4,5,6,7,8,11,14,15,16,17]: hip_r is at index 8, hip_l is at index 9
    HIP_RIGHT_IDX = 8   # position of hip_r in BODY_INDICES
    HIP_LEFT_IDX = 9    # position of hip_l in BODY_INDICES
    
    def __init__(self, mode='train', cfg=None):
        assert mode in ['train', 'test'], f"mode must be 'train' or 'test', got {mode}"
        self.cfg = cfg
        self.mode = mode
        self.data_list = []
        
        # Interpolation settings for short sequences
        self.target_seq_len = getattr(cfg, 'TARGET_SEQ_LEN', 6)
        self.interpolate = getattr(cfg, 'INTERPOLATE_SHORT_SEQ', True)
        
        self._check_dirs()
        self._prepare_splits()
        self._load_split()

    def _check_dirs(self):
        """Verify data directories exist"""
        if not os.path.exists(self.cfg.SKELETON_DIR):
            raise FileNotFoundError(f"Skeleton directory not found: {self.cfg.SKELETON_DIR}")
        if not os.path.exists(self.cfg.PHONO_DIR):
            raise FileNotFoundError(f"Phono directory not found: {self.cfg.PHONO_DIR}")

    def _prepare_splits(self):
        """Create train/test splits if they don't exist"""
        train_file = self.cfg.TRAIN_SPLIT_FILE
        test_file = self.cfg.TEST_SPLIT_FILE
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"Using existing splits: {train_file}, {test_file}")
            return

        print(f"Creating train/test splits from {self.cfg.SKELETON_DIR}...")
        
        # Get all skeleton files
        all_files = [f for f in os.listdir(self.cfg.SKELETON_DIR) if f.endswith('.json')]
        print(f"Found {len(all_files)} skeleton files")
        
        # Group by gloss
        gloss_to_files = defaultdict(list)
        for filename in all_files:
            gloss = self._extract_gloss_from_filename(filename)
            if gloss:
                gloss_to_files[gloss].append(filename)
        
        print(f"Found {len(gloss_to_files)} unique glosses")
        
        # Split: 1 per gloss to test, rest to train
        train_files = []
        test_files = []
        
        for gloss, files in gloss_to_files.items():
            if len(files) == 0:
                continue
            
            # Randomly select one for test
            random.shuffle(files)
            test_files.append(files[0])
            train_files.extend(files[1:])
        
        # Save splits
        os.makedirs(os.path.dirname(train_file) if os.path.dirname(train_file) else '.', exist_ok=True)
        
        with open(train_file, 'w') as f:
            f.write('\n'.join(sorted(train_files)))
        with open(test_file, 'w') as f:
            f.write('\n'.join(sorted(test_files)))
        
        print(f"Splits created - Train: {len(train_files)}, Test: {len(test_files)}")

    def _load_split(self):
        """Load file list from split file"""
        split_file = self.cfg.TRAIN_SPLIT_FILE if self.mode == 'train' else self.cfg.TEST_SPLIT_FILE
        
        with open(split_file, 'r') as f:
            self.data_list = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.data_list)} samples for {self.mode}")

    def _extract_gloss_from_filename(self, filename):
        """
        Extract gloss name from filename.
        Expected format: glossname-123456.json
        Example: "drink-272037.json" -> "drink"
        """
        try:
            # Remove .json extension
            name = filename.replace('.json', '')
            # Split by '-' and take all parts except the last (which is the ID)
            parts = name.rsplit('-', 1)
            if len(parts) >= 1:
                return parts[0].lower()
        except Exception as e:
            print(f"Warning: Could not parse filename {filename}: {e}")
        return None

    def _parse_skeleton_frame(self, frame_data):
        """
        Parse a single frame's skeleton data, extracting only selected joints.
        
        Expected structure:
        {
            "frame_index": int,
            "skeleton": {
                "body": {"x": [...], "y": [...], "z": [...]},
                "face": {"x": [...], "y": [...], "z": [...]},
                "hand_left": {"x": [...], "y": [...], "z": [...]},
                "hand_right": {"x": [...], "y": [...], "z": [...]}
            }
        }
        
        Returns:
            torch.Tensor: (72 * 3,) = (216,) - selected joints only
        """
        skeleton = frame_data.get('skeleton', {})
        coords = []
        
        # 1. Body - extract only upper body joints (14 joints)
        body_data = skeleton.get('body') or {}
        body_xs = body_data.get('x', [])
        body_ys = body_data.get('y', [])
        body_zs = body_data.get('z', [])
        
        for idx in self.BODY_INDICES:
            if idx < len(body_xs) and idx < len(body_ys) and idx < len(body_zs):
                x = body_xs[idx] if body_xs[idx] is not None else 0.0
                y = body_ys[idx] if body_ys[idx] is not None else 0.0
                z = body_zs[idx] if body_zs[idx] is not None else 0.0
                coords.extend([x, y, z])
            else:
                coords.extend([0.0, 0.0, 0.0])
        
        # 2. Face - extract only key landmarks (16 joints)
        face_data = skeleton.get('face') or {}
        face_xs = face_data.get('x', [])
        face_ys = face_data.get('y', [])
        face_zs = face_data.get('z', [])
        
        for idx in self.FACE_INDICES:
            if idx < len(face_xs) and idx < len(face_ys) and idx < len(face_zs):
                x = face_xs[idx] if face_xs[idx] is not None else 0.0
                y = face_ys[idx] if face_ys[idx] is not None else 0.0
                z = face_zs[idx] if face_zs[idx] is not None else 0.0
                coords.extend([x, y, z])
            else:
                coords.extend([0.0, 0.0, 0.0])
        
        # 3. Hand Left - full 21 joints
        hand_left_data = skeleton.get('hand_left') or {}
        hl_xs = hand_left_data.get('x', [])
        hl_ys = hand_left_data.get('y', [])
        hl_zs = hand_left_data.get('z', [])
        
        for i in range(self.HAND_JOINTS):
            if i < len(hl_xs) and i < len(hl_ys) and i < len(hl_zs):
                x = hl_xs[i] if hl_xs[i] is not None else 0.0
                y = hl_ys[i] if hl_ys[i] is not None else 0.0
                z = hl_zs[i] if hl_zs[i] is not None else 0.0
                coords.extend([x, y, z])
            else:
                coords.extend([0.0, 0.0, 0.0])
        
        # 4. Hand Right - full 21 joints
        hand_right_data = skeleton.get('hand_right') or {}
        hr_xs = hand_right_data.get('x', [])
        hr_ys = hand_right_data.get('y', [])
        hr_zs = hand_right_data.get('z', [])
        
        for i in range(self.HAND_JOINTS):
            if i < len(hr_xs) and i < len(hr_ys) and i < len(hr_zs):
                x = hr_xs[i] if hr_xs[i] is not None else 0.0
                y = hr_ys[i] if hr_ys[i] is not None else 0.0
                z = hr_zs[i] if hr_zs[i] is not None else 0.0
                coords.extend([x, y, z])
            else:
                coords.extend([0.0, 0.0, 0.0])
        
        return torch.tensor(coords, dtype=torch.float32)

    def _normalize_pose(self, pose_seq):
        """
        Normalize pose sequence by centering at hip midpoint.
        
        Args:
            pose_seq: (T, 216) - sequence of poses (72 joints * 3)
            
        Returns:
            normalized_seq: (T, 216)
        """
        T = pose_seq.shape[0]
        
        # Reshape to (T, 72, 3) for easier manipulation
        pose_reshaped = pose_seq.view(T, self.TOTAL_JOINTS, 3)
        
        # Hip indices are within the first BODY_JOINTS positions
        # HIP_RIGHT_IDX = 8, HIP_LEFT_IDX = 9 (in our selected body joints)
        hip_right = pose_reshaped[:, self.HIP_RIGHT_IDX, :]  # (T, 3)
        hip_left = pose_reshaped[:, self.HIP_LEFT_IDX, :]    # (T, 3)
        root = (hip_right + hip_left) / 2                    # (T, 3)
        
        # Center all joints
        pose_normalized = pose_reshaped - root.unsqueeze(1)  # (T, 72, 3)
        
        return pose_normalized.view(T, -1)
    
    def _interpolate_sequence(self, pose_seq, target_len):
        """
        Interpolate short sequence to target length using linear interpolation.
        
        Args:
            pose_seq: (T, D) - original short sequence
            target_len: int - desired sequence length
            
        Returns:
            interpolated: (target_len, D) - interpolated sequence
        """
        T, D = pose_seq.shape
        
        if T >= target_len:
            return pose_seq
        
        if T == 1:
            # If only one frame, repeat it
            return pose_seq.repeat(target_len, 1)
        
        # Create interpolation indices
        # Original frames are at positions 0, 1, ..., T-1
        # We want to map them to positions 0, target_len-1 proportionally
        original_indices = torch.linspace(0, target_len - 1, T)
        target_indices = torch.arange(target_len).float()
        
        # Interpolate each dimension
        interpolated = torch.zeros(target_len, D)
        
        for i in range(target_len):
            # Find surrounding original frames
            idx = target_indices[i]
            
            # Find where this index falls in the original sequence
            # Scale to original frame space
            orig_pos = idx * (T - 1) / (target_len - 1)
            
            lower_idx = int(orig_pos)
            upper_idx = min(lower_idx + 1, T - 1)
            
            # Linear interpolation weight
            alpha = orig_pos - lower_idx
            
            interpolated[i] = (1 - alpha) * pose_seq[lower_idx] + alpha * pose_seq[upper_idx]
        
        return interpolated

    def _load_phono(self, skeleton_filename):
        """
        Get gloss label for a sample.
        First try to read from phono file, fallback to filename parsing.
        """
        # Try phono file
        phono_filename = skeleton_filename  # Same filename in different directory
        phono_path = os.path.join(self.cfg.PHONO_DIR, phono_filename)
        
        if os.path.exists(phono_path):
            try:
                with open(phono_path, 'r') as f:
                    phono_data = json.load(f)
                    label = phono_data.get('label') or phono_data.get('gloss', '')
                    if label:
                        # Clean up label (remove # prefix if present)
                        label = label.lstrip('#').lower()
                        return label
            except Exception as e:
                print(f"Warning: Error reading phono file {phono_filename}: {e}")
        

    
    def _get_gloss_label(self, skeleton_data):
        """
        Get gloss label for a sample.
        First try to read from phono file, fallback to filename parsing.
        """

        try:
            label = skeleton_data.get('label') or skeleton_data.get('gloss', '')
            if label:
                # Clean up label (remove # prefix if present)
                label = label.lstrip('#').lower()
                return label
        except Exception as e:
            print(f"Warning: Error reading label from skeleton data: {e}")
            return None
    
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx, debug=False):
        """
        Returns:
            pose_tensor: (T, 216) - normalized pose sequence
            label: str - gloss name
            length: int - actual sequence length (before padding)
        """
        filename = self.data_list[idx]
        skeleton_path = os.path.join(self.cfg.SKELETON_DIR, filename)
                
        # Load skeleton data
        try:
            with open(skeleton_path, 'r') as f:
                data = json.load(f)
                
            label = self._get_gloss_label(data)
            if label is None:
                label = self._extract_gloss_from_filename(filename)
                
        except Exception as e:
            print(f"Error loading skeleton {filename}: {e}")
            # Return dummy data
            dummy_len = self.target_seq_len if self.interpolate else 1
            return torch.zeros(dummy_len, self.cfg.INPUT_DIM), label, dummy_len
        
        # Parse frames
        frames = data.get('frames', [])
        if not frames:
            print(f"Warning: No frames in {filename}")
            dummy_len = self.target_seq_len if self.interpolate else 1
            return torch.zeros(dummy_len, self.cfg.INPUT_DIM), label, dummy_len
        
        num_frames = len(frames)
        # Sort by frame index
        frames = sorted(frames, key=lambda x: x.get('frame_index', 0))
        
        # Parse each frame
        pose_list = []
        for frame in frames:
            pose = self._parse_skeleton_frame(frame)
            pose_list.append(pose)
        
        pose_tensor = torch.stack(pose_list)  # (T, 390)
        
        # Normalize (center at hip)
        pose_tensor = self._normalize_pose(pose_tensor)
        
        # Interpolate if sequence is too short
        if self.interpolate and pose_tensor.shape[0] < self.target_seq_len:
            pose_tensor = self._interpolate_sequence(pose_tensor, self.target_seq_len)
        
        # Truncate if too long
        if pose_tensor.shape[0] > self.cfg.MAX_SEQ_LEN:
            pose_tensor = pose_tensor[:self.cfg.MAX_SEQ_LEN]
        
        length = pose_tensor.shape[0]
        if debug:
            return pose_tensor, label, num_frames
        
        return pose_tensor, label, length

if __name__ == "__main__":
    # Example configuration
    
    class Config:
        
        def __init__(self):
            # ==================== Dataset ====================
            self.DATASET_NAME = "ASLLVD_Skeleton3D"
            self.SKELETON_DIR ='/scratch/rhong5/dataset/ASLLVD/asl-skeleton3d/normalized/3d'
            self.PHONO_DIR = '/scratch/rhong5/dataset/ASLLVD/asl-phono/phonology/3d'
            
            # Split files (auto-generated if not exist)
            self.TRAIN_SPLIT_FILE = "/scratch/rhong5/dataset/ASLLVD/train_split.txt"
            self.TEST_SPLIT_FILE = "/scratch/rhong5/dataset/ASLLVD/test_split.txt"
            
            # ==================== Data Dimensions ====================
            # Upper Body(14) + Face(16) + HandL(21) + HandR(21) = 72 joints
            # 72 * 3 (x,y,z) = 216
            self.INPUT_DIM = 216
            self.MAX_SEQ_LEN = 50
            
            # Sequence interpolation (original data has only 2-4 frames per sample)
            self.TARGET_SEQ_LEN = 5  # Minimum sequence length after interpolation
            self.INTERPOLATE_SHORT_SEQ = True  # Whether to interpolate short sequences
            
            
            # ==================== Model Architecture ====================
            self.LATENT_DIM = 256
            self.MODEL_DIM = 512
            self.N_HEADS = 8
            self.N_LAYERS = 4
            self.DROPOUT = 0.1
            
            # ==================== Training ====================
            self.TRAIN_BSZ = 200
            self.EVAL_BSZ = 200

            
            
            # Curriculum Learning
            self.USE_CURRICULUM = True
            self.MASK_RATIO_MAX = 0.6
            
            # ==================== Hardware ====================
            self.MIXED_PRECISION = "fp16"
            self.NUM_WORKERS = 4
            



    cfg = Config()

    dataset = ASLLVDSkeletonDataset(mode='train', cfg=cfg)
    print(f"Dataset size: {len(dataset)}")
        
    # nan_files = []
    # inf_files = []
    # extreme_files = []

    # nan_txt = "nan_files.txt"
    # with open(nan_txt, 'w') as f_nan:
    #     f_nan.write("Files with NaN values:\n")
    #     for i, dta in enumerate(dataset):
    #         pose_seq, label, length = dta
            
    #         # 检查 NaN
    #         if torch.isnan(pose_seq).any():
    #             nan_count = torch.isnan(pose_seq).sum().item()
    #             nan_files.append((dataset.data_list[i], label, nan_count))
    #             print(f"[{i}] NaN found: {dataset.data_list[i]}, count={nan_count}")
    #             f_nan.write(f"{dataset.data_list[i]}, label={label}, NaN_count={nan_count}\n")
                
            
    #         # 检查 Inf
    #         if torch.isinf(pose_seq).any():
    #             inf_count = torch.isinf(pose_seq).sum().item()
    #             inf_files.append((dataset.data_list[i], label, inf_count))
    #             print(f"[{i}] Inf found: {dataset.data_list[i]}, count={inf_count}")
    #             f_nan.write(f"{dataset.data_list[i]}, label={label}, Inf_count={inf_count}\n")
            
    #         # 检查极端值 (|value| > 100)
    #         max_val = pose_seq.abs().max().item()
    #         if max_val > 100:
    #             extreme_files.append((dataset.data_list[i], label, max_val))
    #             print(f"[{i}] Extreme value: {dataset.data_list[i]}, max={max_val:.2f}")
            
    #         if (i + 1) % 1000 == 0:
    #             print(f"Checked {i+1}/{len(dataset)} samples...")

    # print("\n" + "="*50)
    # print(f"Total samples: {len(dataset)}")
    # print(f"NaN files: {len(nan_files)}") # zero
    # print(f"Inf files: {len(inf_files)}") # zero
    # print(f"Extreme value files (>100): {len(extreme_files)}") # zero 


    from collections import Counter
    
    frame_counts = []
    for i in range(len(dataset)):
        pose_seq, label, orig_frames = dataset.__getitem__(i, debug=True)
        frame_counts.append(orig_frames)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{len(dataset)}...")
    
    # 统计
    counter = Counter(frame_counts)
    total = len(frame_counts)
    
    print("\n帧数分布:")
    print(f"{'帧数':<8} {'数量':<10} {'占比':<10} {'累计':<10}")
    cumulative = 0
    for n in sorted(counter.keys()):
        pct = counter[n] / total * 100
        cumulative += pct
        print(f"{n:<8} {counter[n]:<10} {pct:>5.2f}%     {cumulative:>5.2f}%")
