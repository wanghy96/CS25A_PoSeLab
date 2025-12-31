# app_cli.py
from __future__ import annotations
import argparse
import os
from pose_video_ios_app import process_video_app
from resources import ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--outdir", default="output", help="输出目录（默认 output/）")
    parser.add_argument("--device", default="cpu", help="cpu / cuda:0 / mps")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    process_video_app(
        video_path=args.video,
        out_dir=args.outdir,
        device=args.device,
        kpt_thr=0.3,
    )

if __name__ == "__main__":
    main()