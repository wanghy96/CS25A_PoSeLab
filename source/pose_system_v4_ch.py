import argparse
import numpy as np
import cv2

from mmcv.image import imread
from mmengine.logging import print_log
from mmpose.apis import init_model, inference_topdown
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import logging
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from datetime import datetime

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
# ========== reportlab + macOS 兼容补丁 ==========
import hashlib

try:
    hashlib.md5(usedforsecurity=False)
except TypeError:
    # 重定义一个兼容的 md5，让 reportlab 可以调用
    _old_md5 = hashlib.md5
    def _fixed_md5(*args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _old_md5(*args, **kwargs)
    hashlib.md5 = _fixed_md5
# ===============================================


# ==========================================================
# 0. 工具：角度 & EMA 平滑
# ==========================================================

def angle_between(v1, v2):
    """计算两个向量的夹角（度）"""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def ema_update(prev, current, alpha=0.3):
    """
    指数滑动平均（Exponential Moving Average）
    prev/current 都是 dict（例如 metrics），返回平滑后的 dict。
    目前主程序里不用，你在做视频/实时时可以直接调用。
    """
    if prev is None:
        return current.copy()
    smoothed = {}
    for k, v in current.items():
        smoothed[k] = alpha * v + (1 - alpha) * prev.get(k, v)
    return smoothed


class MetricSmoother:
    """简单封装一个平滑器，在视频/实时中方便使用"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.state = None

    def update(self, metrics: dict):
        self.state = ema_update(self.state, metrics, self.alpha)
        return self.state


# ==========================================================
# 1. 姿态读取：返回 keypoints(17x3), model, img
# ==========================================================

def read_pose(
    img_path: str,
    config: str = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
    checkpoint: str = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
    device: str = 'mps'
):
    """
    读取图片，使用 HRNet 得到 COCO 17 点关键点：
    返回：
        keypoints: (17, 3) [x, y, score]
        model   : mmpose 模型（用于可视化）
        img_bgr : 原图（BGR）
    """
    model = init_model(config, checkpoint, device=device)
    img_bgr = cv2.imread(img_path)

    batch_results = inference_topdown(model, img_bgr)
    data_sample = merge_data_samples(batch_results)

    keypoints_xy = data_sample.pred_instances.keypoints[0]          # (17, 2)
    keypoints_score = data_sample.pred_instances.keypoint_scores[0] # (17,)

    keypoints = np.hstack([keypoints_xy, keypoints_score[:, None]]) # (17, 3)
    return keypoints, model, img_bgr


# ==========================================================
# 2. 医学级姿态分析（v3.0）
# ==========================================================

def analyze_posture_v3(keypoints: np.ndarray) -> dict:
    """
    输入：keypoints (17 x 3)
    输出：一组医学相关的姿态指标：
        FHA：头前倾角 Forward Head Angle
        FHP：头前伸距离 Forward Head Position (px)
        SBA：肩部倾斜角 Shoulder Balance Angle
        PPA：骨盆左右倾斜 Pelvic Position Asymmetry
        PPT：骨盆前后倾 Pelvic Tilt Angle
        TKA：胸椎后凸（驼背） Thoracic Kyphosis Angle (近似)
    """
    xy = keypoints[:, :2]
    sc = keypoints[:, 2]

    nose          = xy[0]
    left_ear      = xy[3]
    right_ear     = xy[4]
    left_shoulder = xy[5]
    right_shoulder= xy[6]
    left_hip      = xy[11]
    right_hip     = xy[12]
    left_knee     = xy[13]
    right_knee    = xy[14]

    # 头部参考点：双耳优先，其次鼻子
    head_pts = []
    if sc[3] > 0.5:
        head_pts.append(left_ear)
    if sc[4] > 0.5:
        head_pts.append(right_ear)
    if not head_pts and sc[0] > 0.5:
        head_pts.append(nose)
    if head_pts:
        head = np.mean(head_pts, axis=0)
    else:
        # 实在没有就退回到肩膀中心，不建议，但为了程序健壮性
        head = (left_shoulder + right_shoulder) / 2

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center      = (left_hip + right_hip) / 2
    knee_center     = (left_knee + right_knee) / 2

    neck_vec  = head - shoulder_center
    spine_vec = shoulder_center - hip_center

    # === 1. FHA：头前倾角（头相对脊柱的屈曲）===
    FHA = angle_between(neck_vec, spine_vec)

    # === 2. FHP：头前伸距离（水平距离，像素）====
    FHP = float(abs(head[0] - shoulder_center[0]))

    # === 3. SBA：肩部倾斜角 ===
    SBA = abs(np.degrees(np.arctan2(
        left_shoulder[1] - right_shoulder[1],
        left_shoulder[0] - right_shoulder[0] + 1e-8
    )))

    # === 4. PPA：骨盆左右倾斜 ===
    PPA = abs(np.degrees(np.arctan2(
        left_hip[1] - right_hip[1],
        left_hip[0] - right_hip[0] + 1e-8
    )))

    # === 5. PPT：骨盆前倾角 ===
    # 用 hip_center → knee_center 与竖直方向的夹角近似
    hip_knee_vec = knee_center - hip_center
    vertical_down = np.array([0, 1.0])
    PPT = angle_between(hip_knee_vec, vertical_down)

    # === 6. TKA：胸椎后凸（驼背）===
    # 用 脊柱向量 与 hip_center → knee_center 向量夹角近似
    TKA = angle_between(spine_vec, (hip_center - knee_center))

    return {
        "FHA": FHA,
        "FHP": FHP,
        "SBA": SBA,
        "PPA": PPA,
        "PPT": PPT,
        "TKA": TKA,
    }


# ==========================================================
# 3. 各项打分函数（保持独立，方便你自己改）
# ==========================================================

import math

def posture_score(x, 
                  t_good=10, t_mid=20, t_bad=40,
                  s_high=100, s_mid=80, s_low=50,
                  low_floor=30):

    if x <= t_good:
        return s_high - (x / t_good) * 2

    elif x <= t_mid:
        return s_high - (s_high - s_mid) * ((x - t_good) / (t_mid - t_good))

    elif x <= t_bad:
        return s_mid - (s_mid - s_low) * ((x - t_mid) / (t_bad - t_mid))

    else:
        return max(low_floor, s_low * math.exp(-(x - t_bad) / 10))


def score_FHA(fha):
    return posture_score(fha, 1, 15, 25, 100, 70, 50, 30)

def score_FHP(fhp):
    return posture_score(fhp, 1, 15, 25, 100, 70, 50, 30)

def score_SBA(sba):
    return posture_score(sba, 5, 10, 15, 100, 80, 60, 40)

def score_PPA(ppa):
    return posture_score(ppa, 5, 10, 15, 100, 80, 60, 40)

def score_PPT(ppt):
    return posture_score(ppt, 10, 20, 30, 100, 80, 65, 45)

def score_TKA(tka):
    return posture_score(tka, 10, 20, 40, 100, 85, 50, 30)


# ==========================================================
# 4. 组合：各部位评分 + 总分
# ==========================================================

def compute_region_scores(metrics: dict) -> dict:
    """
    输入：metrics = analyze_posture_v3 的输出
    输出：包含
        - 单项得分（FHA/FHP/...）
        - 各区域得分（颈椎/胸椎/肩/骨盆）
        - 总分 total_score
    全部放在一个 dict 中，方便你后续扩展。
    """
    fha = metrics["FHA"]
    fhp = metrics["FHP"]
    sba = metrics["SBA"]
    ppa = metrics["PPA"]
    ppt = metrics["PPT"]
    tka = metrics["TKA"]

    fha_score = score_FHA(fha)
    fhp_score = score_FHP(fhp)
    sba_score = score_SBA(sba)
    ppa_score = score_PPA(ppa)
    ppt_score = score_PPT(ppt)
    tka_score = score_TKA(tka)

    # 区域得分（你可以随时在这里改权重）
    cervical_score = 0.6 * fha_score + 0.4 * fhp_score     # 颈椎区
    thoracic_score = tka_score                              # 胸椎区
    shoulder_score = sba_score                              # 肩部
    pelvis_score   = 0.5 * ppa_score + 0.5 * ppt_score      # 骨盆

    # 总分（可随时调整权重）
    total_score = (
    0.45 * cervical_score +   # 颈椎权重提高
    0.35 * thoracic_score +   # 胸椎核心问题
    0.20 * shoulder_score     # 肩部辅助
    )

    return {
        "fha_score": fha_score,
        "fhp_score": fhp_score,
        "sba_score": sba_score,
        "ppa_score": ppa_score,
        "ppt_score": ppt_score,
        "tka_score": tka_score,
        "cervical_score": cervical_score,
        "thoracic_score": thoracic_score,
        "shoulder_score": shoulder_score,
        "pelvis_score": pelvis_score,
        "total_score": total_score,
    }


# ==========================================================
# 5. 可视化：用 MMPose 官方 Visualizer 输出骨架图
# ==========================================================

def visualize_pose_with_mmpose(
    model,
    img_path: str,
    out_path: str,
    draw_heatmap: bool = False,
    kpt_thr: float = 0.3
):
    """
    使用 MMPose 的 Visualizer 输出骨架/heatmap
    独立封装成函数，方便之后单独调用或修改。
    """
    if draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
        # 注意：如果想要 cfg_options 生效，需要在 init_model 时加入，此处为示意。
        # 目前我们用的是已经加载好的 model，所以这里简化处理。
    batch_results = inference_topdown(model, img_path)
    results = merge_data_samples(batch_results)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta, skeleton_style='mmpose')

    img_rgb = imread(img_path, channel_order='rgb')

    visualizer.add_datasample(
        'pose_result',
        img_rgb,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=kpt_thr,
        draw_heatmap=draw_heatmap,
        show_kpt_idx=False,
        show=False,
        out_file=out_path
    )

    print_log(
        f'骨架图已保存到：{out_path}',
        logger='current',
        level=logging.INFO
    )


def detailed_region_report_cn(region: str, score: float, metrics: dict) -> str:
    """根据区域得分与指标生成中文个性化报告"""

    # 严重程度
    if score >= 85:
        severity = (
            "优秀：未发现明显姿态异常。\n"
            "建议：\n"
            "- 保持当前良好的坐姿习惯。\n"
            "- 每 40–60 分钟起身活动，避免颈肩僵硬。\n"
        )
        return severity  # 优秀就不展开问题描述
    elif score >= 70:
        severity = "轻度偏差：存在轻微姿态不平衡。\n"
    elif score >= 50:
        severity = "中度偏差：姿态偏差明显，建议尽快改善。\n"
    else:
        severity = "重度偏差：姿态异常显著，强烈建议进行纠正训练。\n"

    FHA = metrics["FHA"]
    SBA = metrics["SBA"]
    PPA = metrics["PPA"]
    PPT = metrics["PPT"]
    TKA = metrics["TKA"]

    if region == "cervical":  # 颈部
        if FHA > 20:
            detail = (
                "问题：检测到明显的头前倾。\n"
                "- 头前伸会增加颈椎负担，易导致颈部僵硬、头痛、疲劳。\n\n"
                "改善建议：\n"
                "- 进行深颈屈肌训练（如下巴内收）。\n"
                "- 增强胸椎伸展灵活度。\n"
                "- 将屏幕抬至眼睛水平。\n"
            )
        else:
            detail = (
                "问题：存在轻微颈部对齐偏差。\n"
                "改善建议：\n"
                "- 避免长时间低头或身体前倾。\n"
                "- 做轻柔的颈部活动度训练（点头、缓慢旋转）。\n"
            )

    elif region == "thoracic":  # 胸椎
        if TKA > 25:
            detail = (
                "问题：胸椎后凸偏大（含胸、驼背趋势）。\n"
                "改善建议：\n"
                "- 加强竖脊肌与肩胛稳定肌力量。\n"
                "- 伸展胸大肌与胸小肌，减少胸部前侧紧张。\n"
                "- 增加胸椎旋转与伸展灵活度训练。\n"
            )
        else:
            detail = (
                "问题：轻度胸椎僵硬。\n"
                "改善建议：\n"
                "- 进行胸椎旋转与伸展练习（如坐姿绕轴转体）。\n"
                "- 避免长时间含胸低头的坐姿。\n"
            )

    elif region == "shoulder":  # 肩部
        if abs(SBA) > 10:
            detail = (
                "问题：左右肩高度不平衡，提示肩带肌肉用力不对称。\n"
                "改善建议：\n"
                "- 强化下斜方肌、前锯肌等肩胛稳定肌群。\n"
                "- 放松与拉伸上斜方肌，减轻耸肩习惯。\n"
                "- 避免长期单肩背包或偏向一侧的坐姿。\n"
            )
        else:
            detail = (
                "问题：轻微肩胛骨不稳定或姿势不对称。\n"
                "改善建议：\n"
                "- 进行基础肩胛稳定训练（如墙天使、肩胛绕圈）。\n"
                "- 保持胸廓中立，避免无意识耸肩。\n"
            )

    elif region == "pelvis":  # 骨盆
        if abs(PPA) > 10:
            detail = (
                "问题：骨盆侧倾明显，提示左右负重不均衡。\n"
                "改善建议：\n"
                "- 强化臀中肌与躯干侧向稳定肌群。\n"
                "- 伸展腰方肌与腰侧筋膜。\n"
                "- 避免偏向一侧的坐姿或翘腿习惯。\n"
            )
        elif PPT > 25:
            detail = (
                "问题：骨盆前倾偏大，久坐人群常见。\n"
                "改善建议：\n"
                "- 伸展髂腰肌、股直肌等髋屈肌。\n"
                "- 强化臀肌与大腿后侧肌群。\n"
                "- 练习保持脊柱中立位的坐姿与站姿。\n"
            )
        else:
            detail = (
                "问题：存在轻微骨盆姿态偏差。\n"
                "改善建议：\n"
                "- 进行臀肌激活训练（如桥式）。\n"
                "- 增加轻度髋关节灵活度训练。\n"
            )

    return severity + "\n" + detail


# ==========================================================
# 一些pdf小工具：映射名称 / 风险等级 / 评分条
# ==========================================================
def metric_label_map():
    """角度/距离指标的中文说明"""
    return {
        "FHA": "FHA：头前倾角 (°)",
        "FHP": "FHP：头前伸距离 (像素)",
        "SBA": "SBA：肩部倾斜角 (°)",
        "PPA": "PPA：骨盆左右倾斜角 (°)",
        "PPT": "PPT：骨盆前倾角 (°)",
        "TKA": "TKA：胸椎后凸角 (°)",
    }


def region_score_label_map():
    """区域得分"""
    return {
        "cervical_score": "颈部区域",
        "thoracic_score": "胸椎区域",
        "shoulder_score": "肩部区域",
        "pelvis_score": "骨盆区域",
    }


def classify_risk(score: float):
    """根据综合评分给出风险等级与颜色"""
    if score >= 90:
        return "姿态整体优秀", colors.Color(0.1, 0.45, 0.9)
    elif score >= 80:
        return "姿态整体良好", colors.Color(0.2, 0.6, 0.4)
    elif score >= 60:
        return "姿态中等，需要关注", colors.Color(0.9, 0.6, 0.1)
    else:
        return "姿态偏高风险，建议重点改善", colors.Color(0.85, 0.2, 0.2)


def draw_line(c, x1, y1, x2, y2, color=colors.grey, width=0.6):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)


def draw_score_bar(c, x, y, score, label_width=80, bar_width=160, bar_height=8):
    """
    在 (x, y) 位置绘制一行：
    [名称区域] [进度条] [数字]
    调用前请先把名称 drawString 出来
    """
    # 进度条背景
    bar_x = x + label_width
    bar_y = y - 6
    c.setStrokeColor(colors.grey)
    c.setFillColor(colors.white)
    c.setLineWidth(0.6)
    c.roundRect(bar_x, bar_y, bar_width, bar_height, 3, stroke=1, fill=1)

    # 填充根据分数长度
    fill_width = bar_width * max(0, min(score, 100)) / 100.0
    if score >= 85:
        bar_color = colors.Color(0.2, 0.6, 0.4)
    elif score >= 70:
        bar_color = colors.Color(0.9, 0.6, 0.1)
    else:
        bar_color = colors.Color(0.85, 0.2, 0.2)

    c.setFillColor(bar_color)
    c.roundRect(bar_x, bar_y, fill_width, bar_height, 3, stroke=0, fill=1)

    # 分数文字
    c.setFillColor(colors.black)
    c.setFont("STSong-Light", 10)
    c.drawRightString(bar_x + bar_width + 45, y - 1, f"{score:.1f} 分")

def draw_wrapped_text(c, text, x, y, max_width, font_name="STSong-Light", font_size=12, leading=18):
    """
    使用 Paragraph 自动换行绘制文本。
    返回绘制的真实高度，便于动态计算卡片高度。
    """
    style = ParagraphStyle(
        name='CN',
        fontName=font_name,
        fontSize=font_size,
        leading=leading,
        textColor=colors.black,
        alignment=TA_LEFT,
    )
    para = Paragraph(text.replace("\n", "<br/>"), style)
    w, h = para.wrap(max_width, 10000)  # 最大高度给很大，自动计算
    para.drawOn(c, x, y - h)
    return h
# ==========================================================
# PDF 生成模块（美化版）
# ==========================================================
def generate_pdf_report(img_path: str,
                        skel_path: str,
                        metrics: dict,
                        scores: dict,
                        pdf_path: str = "姿态评估报告.pdf"):

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # =======================
    # 页眉
    # =======================
    c.setFillColor(colors.Color(0.90, 0.95, 1.0))
    c.rect(0, height - 90, width, 90, stroke=0, fill=1)

    c.setFillColor(colors.black)
    c.setFont("STSong-Light", 30)
    c.drawString(40, height - 55, "AI 姿态评估报告")

    c.setFont("STSong-Light", 11)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    img_name = img_path.split("/")[-1]
    c.drawRightString(width - 40, height - 55, f"生成时间：{now_str}")
    c.drawString(40, height - 75, f"输入文件：{img_name}")

    draw_line(c, 40, height - 95, width - 40, height - 95)

    # =======================
    # 图片区域
    # =======================
    img_top = height - 120
    img_h, img_w = 240, 220

    # 左图
    c.setFillColor(colors.white)
    c.roundRect(30, img_top - img_h - 40, img_w + 20, img_h + 35, 10, fill=1)
    c.setFont("STSong-Light", 13)
    c.drawString(40, img_top - 20, "原始图像")
    c.drawImage(ImageReader(img_path), 40, img_top - img_h - 10, img_w, img_h)

    # 右图
    c.setFillColor(colors.white)
    c.roundRect(310, img_top - img_h - 40, img_w + 20, img_h + 35, 10, fill=1)
    c.drawString(320, img_top - 20, "关键点检测图")
    c.drawImage(ImageReader(skel_path), 320, img_top - img_h - 10, img_w, img_h)

    # =======================
    # 第一页下方
    # =======================
    bottom = img_top - img_h - 80

    # 左侧关键指标卡片
    c.setFillColor(colors.Color(0.97, 0.97, 0.97))
    c.roundRect(30, bottom - 165, 260, 165, 8, fill=1)

    c.setFillColor(colors.black)
    c.setFont("STSong-Light", 15)
    c.drawString(40, bottom - 20, "关键姿态指标")

    label_map = {
        "FHA": "FHA（头前倾角）：{:.2f}°",
        "FHP": "FHP（头前伸距离）：{:.2f} 像素",
        "SBA": "SBA（肩部倾斜角）：{:.2f}°",
        "PPA": "PPA（骨盆左右倾斜角）：{:.2f}°",
        "PPT": "PPT（骨盆前倾角）：{:.2f}°",
        "TKA": "TKA（胸椎后凸角）：{:.2f}°",
    }

    c.setFont("STSong-Light", 11)
    y = bottom - 45
    for key in ["FHA", "FHP", "SBA", "PPA", "PPT", "TKA"]:
        c.drawString(45, y, label_map[key].format(metrics[key]))
        y -= 18

    # =======================
    # 总体姿态评分卡片（横向布局 + 自动换行）
    # =======================
    card_h = 240
    c.setFillColor(colors.Color(0.97, 0.97, 1.0))
    c.roundRect(320, bottom - card_h, 260, card_h, 8, fill=1)

    total_score = scores["total_score"]
    risk_text, risk_color = classify_risk(total_score)

    # 标题
    c.setFillColor(colors.black)
    c.setFont("STSong-Light", 14)
    c.drawString(330, bottom - 22, "总体姿态评分")

    # 徽章
    badge_x = 330
    badge_y = bottom - 80   # 下移
    badge_w = 110
    badge_h = 45

    c.setFillColor(risk_color)
    c.roundRect(badge_x, badge_y, badge_w, badge_h, 12, fill=1)

    c.setFillColor(colors.white)
    c.setFont("STSong-Light", 20)
    c.drawCentredString(badge_x + badge_w/2, badge_y + 13, f"{total_score:.1f}")

    # 风险描述
    risk_box_x = badge_x + badge_w + 15
    risk_box_y = badge_y + badge_h - 5
    risk_box_width = 260 - badge_w - 40

    draw_wrapped_text(
        c,
        risk_text,
        risk_box_x,
        risk_box_y,
        max_width=risk_box_width,
        font_size=11,
        leading=16
    )

    # 区域评分（进度条已缩短）
    region_labels = region_score_label_map()
    yb = bottom - 130

    short_bar = 120

    for key in ["cervical_score", "thoracic_score", "shoulder_score", "pelvis_score"]:
        score_val = scores[key]
        c.setFillColor(colors.black)
        c.setFont("STSong-Light", 11)
        c.drawString(330, yb, region_labels[key])

        bar_x = 330 + 80
        bar_y = yb - 6

        # 背景条
        c.setFillColor(colors.white)
        c.roundRect(bar_x, bar_y, short_bar, 8, 3, fill=1)

        # 填充
        fill = short_bar * score_val / 100
        if score_val >= 85:
            bar_col = colors.Color(0.2, 0.6, 0.4)
        elif score_val >= 70:
            bar_col = colors.Color(0.9, 0.6, 0.1)
        else:
            bar_col = colors.Color(0.85, 0.2, 0.2)

        c.setFillColor(bar_col)
        c.roundRect(bar_x, bar_y, fill, 8, 3, fill=1, stroke=0)

        # 数字
        c.setFillColor(colors.black)
        c.setFont("STSong-Light", 10)
        c.drawString(bar_x + short_bar + 10, yb - 1, f"{score_val:.1f} 分")

        yb -= 28

    c.showPage()

    # =======================
    # 第二页：分区域分析（Paragraph）
    # =======================
    c.setFont("STSong-Light", 22)
    c.drawString(40, height - 60, "分区域姿态分析与建议")
    draw_line(c, 40, height - 75, width - 40, height - 75)

    y = height - 110
    regions = [
        ("cervical",  "颈部区域", "cervical_score"),
        ("thoracic",  "胸椎区域", "thoracic_score"),
        ("shoulder",  "肩部区域", "shoulder_score"),
        ("pelvis",    "骨盆区域", "pelvis_score"),
    ]

    for reg_key, title, score_key in regions:
        score_val = scores[score_key]

        raw_text = detailed_region_report_cn(reg_key, score_val, metrics)

        # Paragraph 自动换行
        style = ParagraphStyle(
            name='CN2',
            fontName="STSong-Light",
            fontSize=11,
            leading=18,  # APP 风格更好读
            textColor=colors.black,
        )
        para = Paragraph(raw_text.replace("\n", "<br/>"), style)
        avail_width = width - 90  # 两侧留 45
        pw, ph = para.wrap(avail_width, 10000)

        # 不够放 → 换页
        if y - ph - 50 < 40:
            c.showPage()
            c.setFont("STSong-Light", 22)
            c.drawString(40, height - 60, "分区域姿态分析与建议（续）")
            draw_line(c, 40, height - 75, width - 40, height - 75)
            y = height - 110

        # 卡片背景
        card_h = ph + 40
        c.setFillColor(colors.Color(0.97, 0.97, 0.97))
        c.roundRect(30, y - card_h, width - 60, card_h, 10, fill=1)

        # 标题
        c.setFillColor(colors.black)
        c.setFont("STSong-Light", 15)
        c.drawString(40, y - 25, f"{title}：{score_val:.1f} 分")

        # 正文 Paragraph
        para.drawOn(c, 45, y - 45 - ph)

        y -= (card_h + 15)

    c.save()
    print(f"\n✨ 姿态报告已生成：{pdf_path}\n")
# ==========================================================
# 6.（可选）PDF 报告 / 实时摄像头接口：只提供函数框架
# ==========================================================

def generate_pdf_report_stub(img_path: str, metrics: dict, scores: dict, pdf_path: str):
    """
    这里只是一个占位函数（stub），告诉你可以在这里用 reportlab 等库生成 PDF。
    例如：
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        ...
    为避免增加依赖，暂时不强行调用第三方库。
    你之后如果要做启创展示/医工赛的正式报告，可以在这里自己填充。
    """
    pass


def run_webcam_demo_stub():
    """
    实时摄像头 demo 的占位函数。
    你可以在这里：
        - 用 cv2.VideoCapture(0) 读取帧
        - 复用 read_pose / analyze_posture_v3 / compute_region_scores
        - 使用 MetricSmoother 做平滑
        - 在画面上 overlay 分数/角度
    因为你现在更关注单帧图片分析，这里先不展开。
    """
    pass

def run_single_analysis(img_path, pdf_output_path):
    # 1. run pose detection
    keypoints, model, img_bgr = read_pose(img_path)

    # 2. compute metrics
    metrics = analyze_posture_v3(keypoints)

    # 3. compute scores
    scores = compute_region_scores(metrics)

    # 4. generate skeleton
    skel_output = img_path.replace("input_images", "output_skeletons") \
                          .replace(".png", "_skel.jpg") \
                          .replace(".jpg", "_skel.jpg")

    visualize_pose_with_mmpose(model, img_path, skel_output)

    # 5. generate PDF
    generate_pdf_report(img_path, skel_output, metrics, scores, pdf_output_path)
    
# ==========================================================
# 7. 主程序：单张图片流程
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='输入图片路径')
    parser.add_argument(
        '--out',
        default='vis_output.jpg',
        help='骨架可视化输出路径（使用 MMPose Visualizer）'
    )
    args = parser.parse_args()

    # 1) 读取姿态
    keypoints, model, img_bgr = read_pose(args.img)

    # 2) 医学级指标
    metrics = analyze_posture_v3(keypoints)

    # 3) 各部位 & 总分
    scores = compute_region_scores(metrics)

    # ==== 文本输出 ====
    print("\n=== 姿态医学指标（v3.0） ===")
    for k, v in metrics.items():
        print(f"{k:4s} : {v:.2f}")

    print("\n=== 各项评分（0-100） ===")
    print(f"FHA(头前倾)      : {scores['fha_score']:.1f}")
    print(f"FHP(头前伸)      : {scores['fhp_score']:.1f}")
    print(f"SBA(肩部平衡)    : {scores['sba_score']:.1f}")
    print(f"PPA(骨盆左右)    : {scores['ppa_score']:.1f}")
    print(f"PPT(骨盆前倾)    : {scores['ppt_score']:.1f}")
    print(f"TKA(胸椎后凸)    : {scores['tka_score']:.1f}")

    print("\n=== 区域综合评分 ===")
    print(f"颈椎区 Cervical  : {scores['cervical_score']:.1f}")
    print(f"胸椎区 Thoracic  : {scores['thoracic_score']:.1f}")
    print(f"肩部   Shoulder  : {scores['shoulder_score']:.1f}")
    print(f"骨盆   Pelvis    : {scores['pelvis_score']:.1f}")

    print("\n=== 总体坐姿健康评分 ===")
    print(f"Total Score      : {scores['total_score']:.1f} / 100")

    # 4) 输出骨架图（可视化）
    visualize_pose_with_mmpose(model, args.img, args.out, draw_heatmap=False)

    # 5) 生成 PDF 姿态报告
    generate_pdf_report(args.img, args.out, metrics, scores,
                        pdf_path="posture_report.pdf")

if __name__ == '__main__':
    main()