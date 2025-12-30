import cv2
import numpy as np
import streamlit as st
import tempfile
import os
from datetime import datetime
from collections import defaultdict

from utils import get_mediapipe_pose
from frame_instance import FrameInstance
from posture_process import posture_process


class VideoProcessor:
    def __init__(self):
        self.pose = get_mediapipe_pose()
        self.posture_stats = defaultdict(int)
        self.total_frames = 0
        self.good_frames = 0
        
    def process_video(self, video_file, output_path=None):
        """
        处理上传的视频文件，返回处理后的视频路径和统计数据
        """
        # 创建临时文件保存上传的视频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            input_path = tmp_file.name
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 创建输出目录
            output_dir = "output_videos"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"processed_video_{timestamp}.mp4")
        
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("无法打开视频文件，请检查文件格式是否正确")
                return None, None
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 创建视频写入器（使用更兼容的编码器）
            # 尝试多种编码器，确保浏览器兼容性
            fourcc_options = [
                cv2.VideoWriter_fourcc(*'avc1'),  # H.264编码，浏览器兼容性好
                cv2.VideoWriter_fourcc(*'xvid'),  # XVID编码
                cv2.VideoWriter_fourcc(*'mp4v'),  # 原有编码作为后备
                cv2.VideoWriter_fourcc(*'X264'),  # 另一种H.264实现
            ]
            
            out = None
            for fourcc in fourcc_options:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    break
            
            if not out or not out.isOpened():
                st.error("无法创建视频写入器，请检查视频格式")
                return None, None
            
            # 重置统计
            self.posture_stats = defaultdict(int)
            self.total_frames = 0
            self.good_frames = 0
            
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 创建帧实例并处理
                frame_instance = FrameInstance(frame_rgb, self.pose)
                
                if frame_instance.validate():
                    # 进行坐姿检测处理（包括绘制骨骼线和反馈）
                    posture_process(frame_instance, frame_instance.get_frame_width(), frame_instance.get_frame_height())
                    
                    # 分析坐姿并记录统计
                    self._analyze_posture(frame_instance, frame_instance.get_frame_width(), frame_instance.get_frame_height())
                    
                    processed_frame = frame_instance.get_frame()
                else:
                    processed_frame = frame_rgb
                
                # 转换回BGR用于写入
                frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                
                # 更新进度
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f'处理进度: {frame_count}/{total_frames} 帧 ({progress:.1%})')
            
            # 释放资源
            cap.release()
            out.release()
            
            # 清理临时文件
            os.unlink(input_path)
            
            # 清除进度显示
            progress_bar.empty()
            status_text.empty()
            
            # 验证并转换视频格式
            self.validate_and_convert_video(output_path)
            
            return output_path, self._generate_statistics()
            
        except Exception as e:
            st.error(f"视频处理出错: {str(e)}")
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None, None
    
    def validate_and_convert_video(self, output_path):
        """
        验证视频文件并尝试转换为Web兼容格式
        """
        if not os.path.exists(output_path):
            return False, "输出视频文件不存在"
        
        try:
            # 检查视频文件是否可以正常读取
            cap = cv2.VideoCapture(output_path)
            if not cap.isOpened():
                return False, "输出视频文件损坏"
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
                return False, "视频参数无效"
            
            # 创建Web兼容版本
            web_compatible_path = output_path.replace('.mp4', '_web.mp4')
            
            # 使用ffmpeg进行格式转换（如果可用）
            try:
                import subprocess
                cmd = [
                    'ffmpeg', '-i', output_path,
                    '-c:v', 'libx264',    # 使用libx264编码器
                    '-preset', 'fast',       # 快速编码
                    '-crf', '23',          # 质量设置
                    '-c:a', 'aac',         # 音频编码
                    '-y',                  # 覆盖输出文件
                    web_compatible_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # 替换原文件
                os.replace(web_compatible_path, output_path)
                return True, "视频转换完成"
                
            except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
                # 如果ffmpeg不可用，使用OpenCV重新编码
                cap = cv2.VideoCapture(output_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # 使用H.264编码重新创建视频
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(web_compatible_path, fourcc, fps, (width, height))
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                
                cap.release()
                out.release()
                
                # 替换原文件
                os.replace(web_compatible_path, output_path)
                return True, "视频重新编码完成"
                
        except Exception as e:
            return False, f"视频验证失败: {str(e)}"
    
    def _analyze_posture(self, frame_instance, frame_width, frame_height):
        """
        分析坐姿并记录统计数据
        """
        self.total_frames += 1
        
        try:
            # 计算关键角度（使用中心点）
            neck_angle = frame_instance.get_angle('nose', 'shldr_center', 'hip_center')
            spine_angle = frame_instance.get_spine_angle()
            
            # 判断坐姿质量
            neck_good = neck_angle >= 115
            spine_good = spine_angle <= 15
            
            if neck_good and spine_good:
                self.good_frames += 1
                self.posture_stats['良好'] += 1
            else:
                if not neck_good:
                    if neck_angle < 100:
                        self.posture_stats['严重低头'] += 1
                    else:
                        self.posture_stats['轻微低头'] += 1
                
                if not spine_good:
                    if spine_angle > 25:
                        self.posture_stats['严重前倾'] += 1
                    else:
                        self.posture_stats['轻微前倾'] += 1
                        
        except Exception as e:
            # 计算失败，记录为无法检测
            self.posture_stats['无法检测'] += 1
    
    def _generate_statistics(self):
        """
        生成坐姿统计报告
        """
        good_percentage = (self.good_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        
        stats = {
            '总帧数': self.total_frames,
            '良好帧数': self.good_frames,
            '良好比例': f"{good_percentage:.1f}%",
            '详细统计': dict(self.posture_stats)
        }
        
        return stats


def create_video_analysis_ui():
    """
    创建视频分析的用户界面
    """
    st.header("📹 视频坐姿分析")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传视频文件", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="支持 MP4, AVI, MOV, MKV 格式的视频文件"
    )
    
    if uploaded_file is not None:
        # 显示视频信息
        st.video(uploaded_file)
        
        # 处理按钮
        if st.button("🚀 开始分析"):
            with st.spinner("正在处理视频，请稍候..."):
                processor = VideoProcessor()
                output_path, stats = processor.process_video(uploaded_file)
                
                if output_path and stats:                 
                    # 显示分析视频
                    st.subheader("🎥 分析结果视频")
                    st.video(output_path)

                    
                    
                    # 下载链接
                    with open(output_path, "rb") as file:
                        # 从完整路径中提取文件名
                        filename = os.path.basename(output_path)
                        st.download_button(
                            label="下载分析结果视频",
                            data=file.read(),
                            file_name=filename,
                            mime="video/mp4"
                        )
                else:
                    st.error("视频处理失败，请检查视频文件格式和内容。")
    