# pose_video_ios_app.py
from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples

from resources import resource_path, ensure_dir
from plotting import save_metrics_csv, plot_metrics
import json
from collections import Counter

# 从你的核心算法模块导入
from pose_system_v4_ch import (
    analyze_posture_v3,
    compute_region_scores,
    MetricSmoother,
)

# =====================================================
# 0. 字体 & 文本工具（中文 + 自动换行）
# =====================================================

def get_chinese_font(size):
    candidates = [
        resource_path("assets/fonts/NotoSansSC-Regular.ttf"),
        "/System/Library/Fonts/PingFang.ttc",  # mac 开发期兜底
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except:
                pass
    return ImageFont.load_default()

def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int,
                 max_width: int, font: ImageFont.ImageFont,
                 fill=(0, 0, 0), line_spacing=4) -> int:
    if not text:
        return y
    line = ""
    for ch in text:
        if draw.textlength(line + ch, font=font) <= max_width:
            line += ch
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += getattr(font, "size", 18) + line_spacing
            line = ch
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += getattr(font, "size", 18) + line_spacing
    return y


# =====================================================
# 1. 姿态问题识别：给出“问题 + 等级 + 简短描述”
# =====================================================

def _level(x, a, b, c):
    if x < a:
        return "正常"
    if x < b:
        return "轻度"
    if x < c:
        return "中度"
    return "明显"

def detect_posture_problems(metrics: dict, scores: dict):
    FHA = float(metrics.get("FHA", 0.0))
    FHP = float(metrics.get("FHP", 0.0))
    SBA = float(metrics.get("SBA", 0.0))
    PPA = float(metrics.get("PPA", 0.0))
    PPT = float(metrics.get("PPT", 0.0))
    TKA = float(metrics.get("TKA", 0.0))

    problems = []

    lv_fha = _level(FHA, 15, 25, 35)
    if lv_fha != "正常":
        problems.append(("头前倾", lv_fha, f"头前倾角约 {FHA:.1f}°"))

    lv_fhp = _level(FHP, 15, 30, 50)
    if lv_fhp != "正常":
        problems.append(("头前伸", lv_fhp, f"头前伸距离约 {FHP:.1f} 像素"))

    lv_tka = _level(TKA, 20, 30, 40)
    if lv_tka != "正常":
        problems.append(("胸椎后凸 / 驼背", lv_tka, f"胸椎后凸角约 {TKA:.1f}°"))

    lv_sba = _level(SBA, 5, 10, 15)
    if lv_sba != "正常":
        problems.append(("脊柱侧弯趋势", "轻度以上", f"SBA≈{SBA:.1f}°，PPA≈{PPA:.1f}°"))

    # （你原代码里PPT没单列为问题，这里留接口）
    _ = PPT

    if not problems:
        problems.append(("整体姿态", "良好", "当前帧未检测到明显姿态异常。"))

    return problems[:5]


# =====================================================
# 2. 骨架绘制（iOS 风格 + 问题部位高亮）
# =====================================================

COCO_SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

def severity_from_level(level: str):
    if "明显" in level:
        return "severe"
    if "中度" in level:
        return "moderate"
    if "轻度" in level or "以上" in level:
        return "mild"
    if "良好" in level or "正常" in level:
        return "normal"
    return "mild"

def color_for_severity(sev: str):
    if sev == "normal":
        return (89, 199, 52)
    if sev == "mild":
        return (0, 204, 255)
    if sev == "moderate":
        return (0, 149, 255)
    return (48, 59, 255)

def build_highlight_map(problems):
    highlight = {}
    for title, level, _ in problems:
        sev = severity_from_level(level)
        col = color_for_severity(sev)

        if "头前倾" in title or "头前伸" in title:
            highlight["head"] = col
            highlight["neck"] = col
        if "胸椎后凸" in title or "驼背" in title:
            highlight["spine"] = col
        if "骨盆前倾" in title:
            highlight["pelvis"] = col
        if "侧弯" in title:
            highlight["shoulder_line"] = col
            highlight["pelvis_line"] = col
    return highlight

def draw_skeleton_with_highlight(frame, kpt_xy, kpt_score, problems, thr=0.3):
    h, w, _ = frame.shape
    base = max(h, w)
    pt_radius = max(2, base // 300)
    line_thick = max(2, base // 500)

    highlight = build_highlight_map(problems)
    default_line_col = (240, 240, 240)
    default_pt_col = (30, 213, 200)

    left_shoulder, right_shoulder = kpt_xy[5], kpt_xy[6]
    left_hip, right_hip = kpt_xy[11], kpt_xy[12]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    for i, j in COCO_SKELETON:
        if kpt_score[i] < thr or kpt_score[j] < thr:
            continue
        x1, y1 = kpt_xy[i]
        x2, y2 = kpt_xy[j]

        if (i, j) in [(5, 11), (6, 12)] or (j, i) in [(5, 11), (6, 12)]:
            col = highlight.get("spine", default_line_col)
        elif (i, j) in [(11, 13), (12, 14)] or (j, i) in [(11, 13), (12, 14)]:
            col = highlight.get("pelvis", default_line_col)
        else:
            col = default_line_col

        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, line_thick, cv2.LINE_AA)

    if kpt_score[5] > thr and kpt_score[6] > thr:
        col = highlight.get("shoulder_line", default_line_col)
        cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])),
                 (int(right_shoulder[0]), int(right_shoulder[1])),
                 col, line_thick, cv2.LINE_AA)

    if kpt_score[11] > thr and kpt_score[12] > thr:
        col = highlight.get("pelvis_line", highlight.get("pelvis", default_line_col))
        cv2.line(frame, (int(left_hip[0]), int(left_hip[1])),
                 (int(right_hip[0]), int(right_hip[1])),
                 col, line_thick, cv2.LINE_AA)

    col_spine_mid = highlight.get("spine", default_line_col)
    cv2.line(frame, (int(shoulder_center[0]), int(shoulder_center[1])),
             (int(hip_center[0]), int(hip_center[1])),
             col_spine_mid, line_thick, cv2.LINE_AA)

    for idx, (x, y) in enumerate(kpt_xy):
        if kpt_score[idx] < thr:
            continue
        cx, cy = int(x), int(y)

        if idx in [0, 1, 2, 3, 4]:
            col = highlight.get("head", default_pt_col)
        elif idx in [5, 6]:
            col = highlight.get("neck", default_pt_col)
        elif idx in [11, 12, 13, 14, 15, 16]:
            col = highlight.get("pelvis", default_pt_col)
        else:
            col = default_pt_col

        cv2.circle(frame, (cx, cy), pt_radius + 1, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), pt_radius, col, -1)

    return frame


# =====================================================
# 3. iOS 风格信息栏：左侧卡片
# =====================================================

def _level_rgb(level: str):
    sev = severity_from_level(level)
    if sev == "normal":
        return (52, 199, 89)
    if sev == "mild":
        return (255, 204, 0)
    if sev == "moderate":
        return (255, 149, 0)
    return (255, 59, 48)

def compose_ios_frame(frame_bgr, metrics, scores, problems):
    h, w, _ = frame_bgr.shape
    scale = np.sqrt(w * h) / 1500
    scale = max(0.55, min(1.85, scale))

    base_panel = int(240 * scale)
    panel_w = int(max(base_panel, w * 0.25))

    out_w = panel_w + w
    out_h = h

    img = Image.new("RGB", (out_w, out_h), (245, 246, 250))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img.paste(Image.fromarray(frame_rgb), (panel_w, 0))
    draw = ImageDraw.Draw(img)

    def fs(base): return max(10, int(base * scale))

    font_title = get_chinese_font(fs(24))
    font_big = get_chinese_font(fs(22))
    font_mid = get_chinese_font(fs(20))
    font_small = get_chinese_font(fs(18))

    margin = int(18 * scale)
    cx1, cy1 = margin, margin
    cx2, cy2 = panel_w - margin, out_h - margin

    draw.rounded_rectangle(
        [cx1, cy1, cx2, cy2],
        radius=int(26 * scale),
        fill=(255, 255, 255),
        outline=(230, 230, 235),
        width=int(max(2, 2 * scale))
    )

    max_text_width = int((cx2 - cx1) - 40 * scale)

    title_text = "PoseLab 姿态分析"
    if draw.textlength(title_text, font=font_title) > max_text_width:
        title_text = "PoseLab 姿态"

    y = cy1 + int(22 * scale)
    draw.text((cx1 + int(20 * scale), y), title_text, font=font_title, fill=(0, 0, 0))
    y += int(38 * scale)

    total = float(scores.get("total_score", 0.0))
    score_text = f"综合评分：{total:.1f}"
    draw.text((cx1 + int(20 * scale), y), score_text, font=font_big, fill=(10, 90, 220))
    y += int(32 * scale)

    if total >= 90:
        summary = "姿态整体优秀，可持续保持当前习惯。"
    elif total >= 80:
        summary = "姿态整体良好，个别部位可适当关注。"
    elif total >= 60:
        summary = "姿态中等，建议逐步改善学习坐姿。"
    else:
        summary = "姿态偏高风险，建议尽快进行针对性训练。"

    y = draw_wrapped(draw, summary, cx1 + int(20 * scale), y,
                     max_width=max_text_width, font=font_small,
                     fill=(90, 90, 95), line_spacing=int(4 * scale))
    y += int(6 * scale)

    draw.line((cx1 + int(18 * scale), y, cx2 - int(18 * scale), y),
              fill=(230, 230, 235), width=int(max(1, 2 * scale)))
    y += int(20 * scale)

    for title, level, desc in problems:
        if y > cy2 - int(80 * scale):
            break
        color = _level_rgb(level)
        y = draw_wrapped(draw, f"{title}（{level}）",
                         cx1 + int(24 * scale), y,
                         max_width=max_text_width,
                         font=font_mid, fill=color,
                         line_spacing=int(2 * scale))
        y = draw_wrapped(draw, desc,
                         cx1 + int(32 * scale), y,
                         max_width=max_text_width,
                         font=font_small, fill=(100, 100, 110),
                         line_spacing=int(2 * scale))
        y += int(6 * scale)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def save_summary_json(
    records: list[dict],
    out_path: str,
    video_path: str,
    fps: float,
):
    """
    根据 records 汇总生成 summary.json
    """
    # 只统计检测到人体的帧
    valid = [r for r in records if r.get("detected", 0) == 1]

    if not valid:
        summary = {
            "video": os.path.basename(video_path),
            "error": "No valid human detected frames."
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return

    def mean(key):
        return float(np.mean([r[key] for r in valid]))

    scores = [r["total_score"] for r in valid]

    # ---- 主问题统计（基于评分函数逻辑）----
    problem_counter = Counter()
    for r in valid:
        metrics = {
            "FHA": r["FHA"],
            "FHP": r["FHP"],
            "SBA": r["SBA"],
            "PPA": r["PPA"],
            "PPT": r["PPT"],
            "TKA": r["TKA"],
        }
        probs = detect_posture_problems(metrics, {})
        for title, _, _ in probs:
            if title not in ("整体姿态",):
                problem_counter[title] += 1

    summary = {
        "video": os.path.basename(video_path),
        "duration_s": valid[-1]["time_s"],
        "fps": fps,
        "num_frames": len(records),

        "score": {
            "avg": float(np.mean(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },

        "metrics_avg": {
            "FHA": mean("FHA"),
            "FHP": mean("FHP"),
            "SBA": mean("SBA"),
            "PPA": mean("PPA"),
            "PPT": mean("PPT"),
            "TKA": mean("TKA"),
        },

        "main_problems": [
            k for k, _ in problem_counter.most_common(3)
        ],

        "quality": {
            "detected_ratio": len(valid) / len(records)
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

# =====================================================
# 4. 应用化主流程：输出三件套（视频+CSV+图表）
# =====================================================

def process_video_app(
    video_path: str,
    out_dir: str,
    device: str = "mps",
    kpt_thr: float = 0.3,
    config_path: str | None = None,
    checkpoint_path: str | None = None,
    log_cb=None,  # GUI可传入
):
    """
    输入视频，输出：
      - output_video.mp4
      - metrics.csv
      - metrics.png
    """
    ensure_dir(out_dir)

    # 固定资源：默认使用 assets 下的模型
    if config_path is None:
        config_path = resource_path("assets/mmpose_config.py")
    if checkpoint_path is None:
        checkpoint_path = resource_path("assets/mmpose_checkpoint.pth")

    output_video = os.path.join(out_dir, "pose_output.mp4")
    output_csv   = os.path.join(out_dir, "pose_metrics.csv")
    output_png   = os.path.join(out_dir, "pose_metrics.png")
    output_summary = os.path.join(out_dir, "summary.json")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("视频为空或无法读取第一帧。")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    dummy_metrics = {"FHA": 0., "FHP": 0., "SBA": 0., "PPA": 0., "PPT": 0., "TKA": 0.}
    dummy_scores = {"total_score": 100.}
    dummy_probs = [("整体姿态", "正常", "初始化尺寸计算")]

    composed = compose_ios_frame(first_frame.copy(), dummy_metrics, dummy_scores, dummy_probs)
    out_h, out_w, _ = composed.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))

    def log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    log(f"[INFO] 输入尺寸：{W}x{H}，输出尺寸：{out_w}x{out_h}，FPS={fps:.2f}")
    log("[INFO] 初始化 MMPose 模型中 …")
    model = init_model(config_path, checkpoint_path, device=device)

    smoother = MetricSmoother(alpha=0.4)
    frame_idx = 0

    # 时间序列记录（用于CSV和图表）
    records = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        batch_res = inference_topdown(model, frame)
        if len(batch_res) == 0:
            metrics = dummy_metrics
            scores = {"total_score": 0.0}
            probs = [("未检测到人体", "提示", "当前帧未识别到完整人体轮廓。")]
            composed = compose_ios_frame(frame, metrics, scores, probs)
            writer.write(composed)

            records.append({
                "time_s": frame_idx / fps,
                **{k: float(metrics.get(k, 0.0)) for k in ["FHA","FHP","SBA","PPA","PPT","TKA"]},
                "total_score": float(scores.get("total_score", 0.0)),
                "detected": 0,
            })
            continue

        ds = merge_data_samples(batch_res)
        kpt_xy = ds.pred_instances.keypoints[0]
        kpt_score = ds.pred_instances.keypoint_scores[0]

        kpts = np.hstack([kpt_xy, kpt_score[:, None]])
        metrics_raw = analyze_posture_v3(kpts)
        metrics = smoother.update(metrics_raw)
        scores = compute_region_scores(metrics)
        probs = detect_posture_problems(metrics, scores)

        frame_skel = draw_skeleton_with_highlight(frame.copy(), kpt_xy, kpt_score, probs, thr=kpt_thr)
        composed = compose_ios_frame(frame_skel, metrics, scores, probs)
        writer.write(composed)

        # 记录
        records.append({
            "time_s": frame_idx / fps,
            "FHA": float(metrics.get("FHA", 0.0)),
            "FHP": float(metrics.get("FHP", 0.0)),
            "SBA": float(metrics.get("SBA", 0.0)),
            "PPA": float(metrics.get("PPA", 0.0)),
            "PPT": float(metrics.get("PPT", 0.0)),
            "TKA": float(metrics.get("TKA", 0.0)),
            "total_score": float(scores.get("total_score", 0.0)),
            "detected": 1,
        })

        if frame_idx % 20 == 0:
            log(f"[INFO] 已处理 {frame_idx} 帧 …")

    cap.release()
    writer.release()

    # 导出 CSV + 图表
    save_metrics_csv(records, output_csv)
    plot_metrics(records, output_png)
    save_summary_json(
        records,
        output_summary,
        video_path=video_path,
        fps=fps,
    )

    log(
        f"\n✅ 完成：\n"
        f"- 视频：{output_video}\n"
        f"- CSV：{output_csv}\n"
        f"- 图表：{output_png}\n"
        f"- Summary：{output_summary}\n"
    )
    return output_video, output_csv, output_png

