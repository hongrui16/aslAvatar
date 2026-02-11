import os
from pathlib import Path
from collections import Counter

root = Path("/scratch/rhong5/dataset/asl_signbank/smplx_params")  # 改成你的路径

# 1) 统计每个 gloss 文件夹的 npz 数量
gloss_to_n = {}
for p in root.iterdir():
    if not p.is_dir():
        continue
    n = sum(1 for _ in p.glob("*.npz"))
    gloss_to_n[p.name] = n

if not gloss_to_n:
    raise RuntimeError(f"No gloss folders found under: {root.resolve()}")

# 2) 统计“帧数 -> gloss 个数”的分布
dist = Counter(gloss_to_n.values())

num_gloss = len(gloss_to_n)

print(f"Root: {root.resolve()}")
print(f"Total gloss folders: {num_gloss}")
print()

print("frames_per_gloss\tgloss_count\tratio")
for frames, gcount in sorted(dist.items()):  # 按帧数从小到大
    ratio = gcount / num_gloss
    print(f"{frames}\t{gcount}\t{ratio:.4%}")

# 3) （可选）看看哪些 gloss 是某个帧数，比如 0/1/很少/很大
# 例如找出 npz=0 的 gloss
zeros = [g for g, n in gloss_to_n.items() if n == 0]
if zeros:
    print("\nGloss with 0 npz (first 30):")
    print(zeros[:30])

# 4) （可选）保存两个 CSV：分布 + 每个 gloss 的帧数
# 分布
dist_csv = root / "npz_count_distribution.csv"
with dist_csv.open("w", encoding="utf-8") as f:
    f.write("frames_per_gloss,gloss_count,ratio\n")
    for frames, gcount in sorted(dist.items()):
        f.write(f"{frames},{gcount},{gcount/num_gloss:.8f}\n")

# 每个 gloss 的计数
per_gloss_csv = root / "npz_count_per_gloss.csv"
with per_gloss_csv.open("w", encoding="utf-8") as f:
    f.write("gloss,npz_count\n")
    for gloss, n in sorted(gloss_to_n.items(), key=lambda x: (-x[1], x[0])):
        f.write(f"{gloss},{n}\n")

print(f"\nSaved: {dist_csv}")
print(f"Saved: {per_gloss_csv}")
