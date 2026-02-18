import torch
import numpy as np



# ============================================================================
# Joint Selection Constants
# ============================================================================

# Full 53-joint array indices
ALL_53_JOINTS = list(range(53))

# Lower body + jaw indices to REMOVE (8 joints)
LOWER_BODY_INDICES = [1, 2, 4, 5, 7, 8, 10, 11]  # hips, knees, ankles, feet
REMOVE_INDICES = set(LOWER_BODY_INDICES)  # 8 joints

# Upper body indices to KEEP (44 joints)
UPPER_BODY_INDICES = [i for i in ALL_53_JOINTS if i not in REMOVE_INDICES]
assert len(UPPER_BODY_INDICES) == 45, f"Expected 45 upper body joints, got {len(UPPER_BODY_INDICES)}"

# Joint names for reference / debugging
FULL_JOINT_NAMES = {
    0: 'root/pelvis',
    1: 'left_hip', 2: 'right_hip', 3: 'spine1',
    4: 'left_knee', 5: 'right_knee', 6: 'spine2',
    7: 'left_ankle', 8: 'right_ankle', 9: 'spine3',
    10: 'left_foot', 11: 'right_foot', 12: 'neck',
    13: 'left_collar', 14: 'right_collar', 15: 'head',
    16: 'left_shoulder', 17: 'right_shoulder',
    18: 'left_elbow', 19: 'right_elbow',
    20: 'left_wrist', 21: 'right_wrist',
    # 22-36: left hand fingers (15 joints)
    # 37-51: right hand fingers (15 joints)
    52: 'jaw',
}
for i in range(15):
    FULL_JOINT_NAMES[22 + i] = f'lhand_{i}'
    FULL_JOINT_NAMES[37 + i] = f'rhand_{i}'


# ============================================================================
# Rotation Conversion Utilities
# ============================================================================

def axis_angle_to_matrix(rotvec):
    """
    Convert axis-angle to rotation matrix using Rodrigues' formula.

    Args:
        rotvec: (*, 3) axis-angle vectors

    Returns:
        matrix: (*, 3, 3) rotation matrices
    """
    shape = rotvec.shape[:-1]
    angle = torch.norm(rotvec, dim=-1, keepdim=True)  # (*, 1)

    # Handle zero-angle case
    near_zero = (angle.squeeze(-1) < 1e-6)
    
    # Safe normalize
    axis = rotvec / (angle + 1e-8)  # (*, 3)

    cos_a = torch.cos(angle).unsqueeze(-1)  # (*, 1, 1)
    sin_a = torch.sin(angle).unsqueeze(-1)  # (*, 1, 1)

    # Skew-symmetric matrix K
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=-1).reshape(*shape, 3, 3)

    # Rodrigues: R = I + sin(a)*K + (1-cos(a))*K^2
    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(*shape, 3, 3)
    R = I + sin_a * K + (1 - cos_a) * (K @ K)

    # For near-zero angles, use identity
    if near_zero.any():
        R[near_zero] = I[near_zero]

    return R


def matrix_to_rot6d(matrix):
    """
    Convert rotation matrix to 6D representation (first two columns).

    Args:
        matrix: (*, 3, 3) rotation matrices

    Returns:
        rot6d: (*, 6) 6D rotation vectors
    """
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def rot6d_to_matrix(rot6d):
    """
    Convert 6D rotation back to rotation matrix via Gram-Schmidt.

    Args:
        rot6d: (*, 6) 6D rotation vectors

    Returns:
        matrix: (*, 3, 3) rotation matrices
    """
    shape = rot6d.shape[:-1]
    a1 = rot6d[..., :3]  # first column
    a2 = rot6d[..., 3:]  # second column

    # Gram-Schmidt orthogonalization
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (*, 3, 3)


def matrix_to_axis_angle(matrix):
    """
    Convert rotation matrix to axis-angle.

    Args:
        matrix: (*, 3, 3) rotation matrices

    Returns:
        rotvec: (*, 3) axis-angle vectors
    """
    # Use the relationship: trace(R) = 1 + 2*cos(angle)
    # and the skew-symmetric part for the axis
    batch_shape = matrix.shape[:-2]

    # Angle from trace
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)  # (*)

    # Axis from skew-symmetric part: [R - R^T] / (2 sin(angle))
    skew = matrix - matrix.transpose(-2, -1)  # (*, 3, 3)
    axis = torch.stack([
        skew[..., 2, 1],
        skew[..., 0, 2],
        skew[..., 1, 0]
    ], dim=-1)  # (*, 3)

    sin_angle = torch.sin(angle).unsqueeze(-1)  # (*, 1)
    near_zero = (angle < 1e-6).unsqueeze(-1)

    # Normalize axis
    axis = axis / (2.0 * sin_angle + 1e-8)
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)

    rotvec = axis * angle.unsqueeze(-1)

    # Near-zero angle: return zeros
    rotvec = torch.where(near_zero.expand_as(rotvec), torch.zeros_like(rotvec), rotvec)

    return rotvec


def axis_angle_to_rot6d(rotvec):
    """Shortcut: axis-angle → 6D rotation."""
    return matrix_to_rot6d(axis_angle_to_matrix(rotvec))


def rot6d_to_axis_angle(rot6d):
    """Shortcut: 6D rotation → axis-angle."""
    return matrix_to_axis_angle(rot6d_to_matrix(rot6d))






def postprocess_motion(motion_raw, cfg):
    """
    Convert model output back to full SMPL-X axis-angle (T, 159).

    Args:
        motion_raw: np.ndarray (T, input_dim) — raw model output
                    e.g. (T, 264) if rot6d + upper_body
        cfg: config with USE_ROT6D, USE_UPPER_BODY

    Returns:
        np.ndarray (T, 159) — 53 joints × 3 axis-angle
    """
    use_rot6d = getattr(cfg, 'USE_ROT6D', False)
    use_upper_body = getattr(cfg, 'USE_UPPER_BODY', False)

    seq = torch.from_numpy(motion_raw).float()
    T = seq.shape[0]

    n_joints = 44 if use_upper_body else 53
    n_feats = 6 if use_rot6d else 3

    # (T, input_dim) → (T, N_joints, N_feats)
    seq = seq.reshape(T, n_joints, n_feats)

    # 6D → axis-angle
    if use_rot6d:
        seq = rot6d_to_axis_angle(seq)  # (T, N_joints, 3)

    # 44 upper body → 53 full body (lower body = zeros = neutral pose)
    if use_upper_body:
        full = torch.zeros(T, 53, 3, dtype=seq.dtype)
        for new_idx, orig_idx in enumerate(UPPER_BODY_INDICES):
            full[:, orig_idx, :] = seq[:, new_idx, :]
        seq = full

    return seq.reshape(T, -1).numpy()  # (T, 159)