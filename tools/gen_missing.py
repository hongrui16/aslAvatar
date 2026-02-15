"""
gen_missing.py — 生成 missing.txt 提交给 WLASL 作者申请缺失视频

用法:
  python gen_missing.py \
      --wlasl_json /scratch/rhong5/dataset/wlasl/WLASL_v0.3.json \
      --video_dir /scratch/rhong5/dataset/wlasl/videos/ \
      --output missing.txt
"""

import os, json, argparse

def find_video(video_dir, video_id):
    for ext in ('.mp4', '.avi', '.mov', '.mkv'):
        if os.path.exists(os.path.join(video_dir, f"{video_id}{ext}")):
            return True
    return False

parser = argparse.ArgumentParser()
parser.add_argument('--wlasl_json', type=str, required=True)
parser.add_argument('--video_dir', type=str, required=True)
parser.add_argument('--output', type=str, default='missing.txt')
args = parser.parse_args()

with open(args.wlasl_json) as f:
    wlasl = json.load(f)

missing = []
found = 0
for entry in wlasl:
    for inst in entry.get('instances', []):
        vid = inst.get('video_id', '')
        if vid and not find_video(args.video_dir, vid):
            missing.append(vid)
        else:
            found += 1

with open(args.output, 'w') as f:
    for vid in missing:
        f.write(vid + '\n')

print(f"Found: {found}, Missing: {len(missing)}")
print(f"Saved: {args.output}")
