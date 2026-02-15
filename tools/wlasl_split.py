"""
wlasl_split.py — 筛选 WLASL glosses, 输出 train.txt / test.txt

输出格式 (每行):
  gloss /absolute/path/to/video.mp4

用法:
  python wlasl_split.py \
      --wlasl_json /path/to/WLASL_v0.3.json \
      --video_dir /path/to/wlasl_videos \
      --output_dir ./wlasl_processed \
      --min_samples 18 \
      --test_per_gloss 2
"""

import os
import json
import argparse
import random


def find_video(video_dir, video_id, gloss=None):
    """在 video_dir 下找视频文件，返回绝对路径或 None"""
    for ext in ('.mp4', '.avi', '.mov', '.mkv'):
        # 常见: video_dir/VIDEO_ID.mp4
        p = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(p):
            return os.path.abspath(p)
        # 有些版本按 gloss 分目录: video_dir/gloss/VIDEO_ID.mp4
        if gloss:
            p = os.path.join(video_dir, gloss, f"{video_id}{ext}")
            if os.path.exists(p):
                return os.path.abspath(p)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wlasl_json', type=str, required=True,
                        help='WLASL_v0.3.json 路径')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='WLASL 视频目录')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='输出目录 (train.txt, test.txt)')
    parser.add_argument('--min_samples', type=int, default=18,
                        help='每个 gloss 最少视频数')
    parser.add_argument('--test_per_gloss', type=int, default=2,
                        help='每个 gloss 分给 test 的视频数')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- 加载 JSON ----
    with open(args.wlasl_json, 'r') as f:
        wlasl = json.load(f)
    print(f"WLASL JSON: {len(wlasl)} glosses")

    # ---- 筛选: 找到视频文件存在且 >= min_samples 的 glosses ----
    qualified = []  # [(gloss, [(video_id, path), ...]), ...]
    total_found = 0
    total_missing = 0

    for entry in wlasl:
        gloss = entry['gloss']
        found = []
        for inst in entry.get('instances', []):
            vid = inst.get('video_id', '')
            path = find_video(args.video_dir, vid, gloss)
            if path:
                found.append((vid, path))
            else:
                total_missing += 1
        total_found += len(found)

        if len(found) >= args.min_samples:
            qualified.append((gloss, found))

    # 按样本数降序
    qualified.sort(key=lambda x: len(x[1]), reverse=True)

    print(f"Videos found: {total_found}, missing: {total_missing}")
    print(f"Qualified glosses (>= {args.min_samples}): {len(qualified)}")
    print(f"(论文: 103 glosses)")

    # ---- Split ----
    train_lines = []
    test_lines = []

    for gloss, videos in qualified:
        random.shuffle(videos)
        n_test = min(args.test_per_gloss, len(videos) - 1)  # 至少留 1 个 train
        test_vids = videos[:n_test]
        train_vids = videos[n_test:]

        for vid, path in train_vids:
            train_lines.append(f"{gloss} {path}")
        for vid, path in test_vids:
            test_lines.append(f"{gloss} {path}")

    # ---- 写文件 ----
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, 'train.txt')
    test_path = os.path.join(args.output_dir, 'test.txt')

    with open(train_path, 'w') as f:
        f.write('\n'.join(train_lines) + '\n')
    with open(test_path, 'w') as f:
        f.write('\n'.join(test_lines) + '\n')

    print(f"\ntrain.txt: {len(train_lines)} samples")
    print(f"test.txt:  {len(test_lines)} samples")
    print(f"Saved to: {args.output_dir}/")

    # ---- 打印 gloss 列表 ----
    print(f"\n{'#':>3} {'Gloss':<20} {'Train':>5} {'Test':>5} {'Total':>5}")
    print("-" * 45)
    for i, (gloss, videos) in enumerate(qualified):
        n_test = min(args.test_per_gloss, len(videos) - 1)
        n_train = len(videos) - n_test
        print(f"{i+1:3d} {gloss:<20} {n_train:5d} {n_test:5d} {len(videos):5d}")

    print(f"\n总计: {len(qualified)} glosses, "
          f"{len(train_lines)} train, {len(test_lines)} test")


if __name__ == '__main__':
    main()
