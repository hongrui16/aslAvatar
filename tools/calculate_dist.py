import os
from pathlib import Path
from collections import Counter

def main_asl_signbank():
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
    dist_csv = "npz_count_distribution.csv"
    with dist_csv.open("w", encoding="utf-8") as f:
        f.write("frames_per_gloss,gloss_count,ratio\n")
        for frames, gcount in sorted(dist.items()):
            f.write(f"{frames},{gcount},{gcount/num_gloss:.8f}\n")

    # 每个 gloss 的计数
    per_gloss_csv = "npz_count_per_gloss.csv"
    with per_gloss_csv.open("w", encoding="utf-8") as f:
        f.write("gloss,npz_count\n")
        for gloss, n in sorted(gloss_to_n.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{gloss},{n}\n")

    print(f"\nSaved: {dist_csv}")
    print(f"Saved: {per_gloss_csv}")


def main_wlasl():
    # root = Path("/scratch/rhong5/dataset/wlasl/train/smplx_params")
    root = Path("/scratch/rhong5/dataset/wlasl/test/smplx_params")

    frame_counts = []  # 每个视频的帧数

    for gloss_dir in sorted(root.iterdir()):
        if not gloss_dir.is_dir():
            continue
        # 每个子目录 = 一个视频
        for video_dir in sorted(gloss_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            n = sum(1 for _ in video_dir.glob("*.npz"))
            frame_counts.append(n)

    s = sorted(frame_counts)
    n = len(s)
    median = s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2
    p10, p25, p75, p90 = s[n//10], s[n//4], s[3*n//4], s[9*n//10]

    print(f"Total videos: {n}")
    print(f"Total frames: {sum(s)}")
    print(f"Min: {min(s)}, Max: {max(s)}, Mean: {sum(s)/n:.1f}, Median: {median}")
    print(f"P10: {p10}, P25: {p25}, P75: {p75}, P90: {p90}")
    print(f"50% of videos fall in [{p25}, {p75}] frames")
    print(f"80% of videos fall in [{p10}, {p90}] frames")
    print()

    dist = Counter(frame_counts)
    print("frames\tvideos\tratio")
    for f, c in sorted(dist.items()):
        print(f"{f}\t{c}\t{c/len(frame_counts):.2%}")




def find_low_frame():
    root = Path("/scratch/rhong5/dataset/wlasl/train/smplx_params")
    min_thresh = 2  # 低于这个帧数的视频列出来

    low = []
    total = 0
    for gloss_dir in sorted(root.iterdir()):
        if not gloss_dir.is_dir():
            continue
        for video_dir in sorted(gloss_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            n = sum(1 for _ in video_dir.glob("*.npz"))
            total += 1
            if n < min_thresh:
                low.append((gloss_dir.name, video_dir.name, n))

    print(f"Total videos: {total}")
    print(f"Videos with < {min_thresh} frames: {len(low)} ({len(low)/total:.1%})")
    print(f"\n{'Gloss':<20} {'Video':<30} {'Frames':>6}")
    print("-" * 58)
    for g, v, n in sorted(low, key=lambda x: x[2]):
        print(f"{g:<20} {v:<30} {n:>6}")
    
# find_low_frame()
main_wlasl()