"""
wlasl_diag.py — 看看每个 gloss 实际有多少视频可用
"""
import os, json, argparse

def find_video(video_dir, video_id, gloss=None):
    for ext in ('.mp4', '.avi', '.mov', '.mkv'):
        p = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(p):
            return p
        if gloss:
            p = os.path.join(video_dir, gloss, f"{video_id}{ext}")
            if os.path.exists(p):
                return p
    return None

parser = argparse.ArgumentParser()
parser.add_argument('--wlasl_json', type=str, required=True)
parser.add_argument('--video_dir', type=str, required=True)
args = parser.parse_args()

with open(args.wlasl_json) as f:
    wlasl = json.load(f)

# 1. 看看视频目录里到底有什么
video_files = os.listdir(args.video_dir)
print(f"视频目录文件数: {len(video_files)}")
print(f"前 10 个文件: {sorted(video_files)[:10]}")
print()

# 2. 看看 JSON 里 video_id 长什么样
sample_entry = wlasl[0]
print(f"JSON 第一个 gloss: '{sample_entry['gloss']}'")
print(f"  instances 数: {len(sample_entry.get('instances', []))}")
if sample_entry.get('instances'):
    inst = sample_entry['instances'][0]
    print(f"  第一个 instance keys: {list(inst.keys())}")
    print(f"  video_id: '{inst.get('video_id', '')}'")
    print(f"  url: '{inst.get('url', '')[:80]}'")
print()

# 3. 统计每个 gloss 的 (标注数, 实际找到数)
counts = []
for entry in wlasl:
    gloss = entry['gloss']
    instances = entry.get('instances', [])
    n_annotated = len(instances)
    n_found = 0
    for inst in instances:
        vid = inst.get('video_id', '')
        if find_video(args.video_dir, vid, gloss):
            n_found += 1
    counts.append((gloss, n_annotated, n_found))

counts.sort(key=lambda x: x[2], reverse=True)

# 4. 打印 top 30
print(f"{'#':>3} {'Gloss':<20} {'Annotated':>9} {'Found':>6}")
print("-" * 42)
for i, (g, na, nf) in enumerate(counts[:30]):
    print(f"{i+1:3d} {g:<20} {na:9d} {nf:6d}")

# 5. 分布
thresholds = [1, 3, 5, 8, 10, 12, 15, 18, 20]
print(f"\n可用视频数 >= N 的 gloss 数量:")
for t in thresholds:
    n = sum(1 for _, _, nf in counts if nf >= t)
    total_vids = sum(nf for _, _, nf in counts if nf >= t)
    print(f"  >= {t:2d}: {n:4d} glosses, {total_vids:5d} videos")
