import numpy as np
import os
from collections import Counter



def compare_wlasl_smplx_param_inside_one_gloss():
    smplx_params_dir = '/scratch/rhong5/dataset/wlasl/train/smplx_params/'


    # 统计每个gloss的视频数量
    gloss_counts = {}
    for gloss_dir in os.listdir(smplx_params_dir):
        gloss_path = os.path.join(smplx_params_dir, gloss_dir)
        if not os.path.isdir(gloss_path):
            continue
        n_videos = sum(1 for v in os.listdir(gloss_path) if os.path.isdir(os.path.join(gloss_path, v)))
        gloss_counts[gloss_dir] = n_videos

    # 取视频数最多的5个
    top5 = sorted(gloss_counts, key=gloss_counts.get, reverse=True)[:5]
    print("Top 5 glosses:", [(g, gloss_counts[g]) for g in top5])

    for gloss_dir in top5:
        all_body, all_lhand, all_rhand, all_root = [], [], [], []
        gloss_path = os.path.join(smplx_params_dir, gloss_dir)
        for vid in sorted(os.listdir(gloss_path)):
            frames_dir = os.path.join(gloss_path, vid)
            if not os.path.isdir(frames_dir):
                continue
            for f in sorted(os.listdir(frames_dir)):
                if not f.endswith('.npz'):
                    continue
                data = np.load(os.path.join(frames_dir, f), allow_pickle=True)
                all_root.append(data.get('smplx_root_pose', np.zeros(3)).flatten())
                all_body.append(data.get('smplx_body_pose', np.zeros(63)).flatten())
                all_lhand.append(data.get('smplx_lhand_pose', np.zeros(45)).flatten())
                all_rhand.append(data.get('smplx_rhand_pose', np.zeros(45)).flatten())
        
        root = np.stack(all_root)
        body = np.stack(all_body)
        lhand = np.stack(all_lhand)
        rhand = np.stack(all_rhand)
        
        print(f"\n=== {gloss_dir} ({len(root)} frames) ===")
        print(f"  root:  mean={np.abs(root).mean():.4f}, std={root.std():.4f}, range={root.ptp(axis=0).mean():.4f}")
        print(f"  body:  mean={np.abs(body).mean():.4f}, std={body.std():.4f}, range={body.ptp(axis=0).mean():.4f}")
        print(f"  lhand: mean={np.abs(lhand).mean():.4f}, std={lhand.std():.4f}, range={lhand.ptp(axis=0).mean():.4f}")
        print(f"  rhand: mean={np.abs(rhand).mean():.4f}, std={rhand.std():.4f}, range={rhand.ptp(axis=0).mean():.4f}")
        
        


def compare_wlasl_smplx_param_inside_among_glosses():
    smplx_params_dir = '/scratch/rhong5/dataset/wlasl/train/smplx_params/'


    # 统计每个gloss的视频数量
    gloss_counts = {}
    for gloss_dir in os.listdir(smplx_params_dir):
        gloss_path = os.path.join(smplx_params_dir, gloss_dir)
        if not os.path.isdir(gloss_path):
            continue
        n_videos = sum(1 for v in os.listdir(gloss_path) if os.path.isdir(os.path.join(gloss_path, v)))
        gloss_counts[gloss_dir] = n_videos

    # 取视频数最多的5个
    top5 = sorted(gloss_counts, key=gloss_counts.get, reverse=True)[:5]
    print("Top 5 glosses:", [(g, gloss_counts[g]) for g in top5])

    gloss_means = {}
    for gloss_dir in top5:
        all_body, all_lhand, all_rhand = [], [], []
        gloss_path = os.path.join(smplx_params_dir, gloss_dir)
        for vid in sorted(os.listdir(gloss_path)):
            frames_dir = os.path.join(gloss_path, vid)
            if not os.path.isdir(frames_dir):
                continue
            for f in sorted(os.listdir(frames_dir)):
                if not f.endswith('.npz'):
                    continue
                data = np.load(os.path.join(frames_dir, f), allow_pickle=True)
                all_body.append(data.get('smplx_body_pose', np.zeros(63)).flatten())
                all_lhand.append(data.get('smplx_lhand_pose', np.zeros(45)).flatten())
                all_rhand.append(data.get('smplx_rhand_pose', np.zeros(45)).flatten())
        
        gloss_means[gloss_dir] = {
            'body': np.stack(all_body).mean(axis=0),
            'lhand': np.stack(all_lhand).mean(axis=0),
            'rhand': np.stack(all_rhand).mean(axis=0),
        }

    # 跨gloss两两比较
    print("=== Cross-gloss L2 distance (mean pose) ===")
    for part in ['body', 'lhand', 'rhand']:
        print(f"\n{part}:")
        glosses = list(gloss_means.keys())
        for i in range(len(glosses)):
            for j in range(i+1, len(glosses)):
                dist = np.linalg.norm(gloss_means[glosses[i]][part] - gloss_means[glosses[j]][part])
                print(f"  {glosses[i]:8s} vs {glosses[j]:8s}: {dist:.4f}")
                
# compare_wlasl_smplx_param_inside_among_glosses()
compare_wlasl_smplx_param_inside_one_gloss()