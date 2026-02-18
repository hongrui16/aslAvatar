"""
gen_retry.py — 找出帧数 < N 的视频, 生成 retry.txt 重跑

用法:
  python gen_retry.py \
      --smplx_dir /scratch/rhong5/dataset/wlasl/train/smplx_params \
      --orig_txt /scratch/rhong5/dataset/wlasl/train.txt \
      --min_frames 5 \
      --output retry.txt
"""
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--smplx_dir', type=str, required=True)
parser.add_argument('--orig_txt', type=str, required=True)
parser.add_argument('--min_frames', type=int, default=5)
parser.add_argument('--output', type=str, default='retry.txt')
args = parser.parse_args()

# 读原始 txt
lines = {}
with open(args.orig_txt) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            gloss, video_path = parts[0], parts[1]
            video_name = Path(video_path).stem
            lines[(gloss, video_name)] = line.strip()

# 检查每个视频的帧数
root = Path(args.smplx_dir)
retry = []
ok = 0

for gloss_dir in sorted(root.iterdir()):
    if not gloss_dir.is_dir():
        continue
    gloss = gloss_dir.name
    for video_dir in sorted(gloss_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        n = sum(1 for _ in video_dir.glob("*.npz"))
        if n < args.min_frames:
            key = (gloss, video_dir.name)
            if key in lines:
                retry.append(lines[key])
            else:
                print(f"  WARN: {gloss}/{video_dir.name} ({n} frames) not found in txt")
        else:
            ok += 1

with open(args.output, 'w') as f:
    f.write('\n'.join(retry) + '\n')

print(f"OK (>= {args.min_frames} frames): {ok}")
print(f"Retry (< {args.min_frames} frames): {len(retry)}")
print(f"Saved: {args.output}")
