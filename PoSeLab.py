import av
import os
import sys
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder
import traceback
from datetime import datetime


BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)


from utils import get_mediapipe_pose
from posture_process import posture_process
from frame_instance import FrameInstance
from video_processor import create_video_analysis_ui

st.title('🧍 AI坐姿检测系统')

# 创建选项卡
tab1, tab2 = st.tabs(["📹 实时检测", "📁 视频分析"])

# 实时检测选项卡
with tab1:
    st.header("📹 实时坐姿检测")
    
    # Initialize pose solution
    pose = get_mediapipe_pose()

    # 初始化会话状态
    if 'download' not in st.session_state:
        st.session_state['download'] = False

    # 创建输出目录
    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用固定的输出文件名
    output_video_file = os.path.join(output_dir, 'output_live.flv')

    def video_frame_callback(frame: av.VideoFrame):
        try:
            frame = frame.to_ndarray(format="rgb24")  # Decode and get RGB frame
            frame_instance = FrameInstance(frame, pose)  # Create frame instance
            
            if frame_instance.validate():
                # Process posture detection
                posture_process(frame_instance, frame_instance.get_frame_width(), frame_instance.get_frame_height())
                processed_frame = frame_instance.get_frame()
                return av.VideoFrame.from_ndarray(processed_frame, format="rgb24")  # Encode and return frame
            else:
                # No person detected, return original frame
                return av.VideoFrame.from_ndarray(frame, format="rgb24")
        except Exception as ex:
            print(f"视频帧处理错误: {str(ex)}")
            traceback.print_exc()
            return frame  # Return original frame if error occurs

    def out_recorder_factory() -> MediaRecorder:
            return MediaRecorder(output_video_file, format="flv")

    ctx = webrtc_streamer(
                            key="posture-detection",
                            video_frame_callback=video_frame_callback,
                            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                            media_stream_constraints={"video": {"width": {'min':720, 'ideal':720}}, "audio": False},
                            video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True),
                            out_recorder_factory=out_recorder_factory
                        )

# 视频分析选项卡
with tab2:
    create_video_analysis_ui()
