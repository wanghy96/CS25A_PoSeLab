from frame_instance import FrameInstance
import threading


def posture_process(frame_instance, frame_width, frame_height):
    """
    坐姿检测核心处理函数 (3D版本)
    检测脖子前倾角度和脊柱弯曲角度，使用3D坐标提高鲁棒性
    """
    
    # 使用3D坐标直接进行坐姿检测，无需朝向判断
    
    # 绘制脊柱线（肩中心到髋中心）
    frame_instance.draw_spine_line()
    
    # 1. 检测脖子前倾角度（鼻-肩中心-髋中心3D夹角）
    neck_angle = frame_instance.get_angle_and_draw('nose', 'shldr_center', 'hip_center', 
                                                   text_color='light_green', 
                                                   line_color='light_blue', 
                                                   point_color='yellow')
    
    # 2. 检测脊柱弯曲角度（肩中心-髋中心-垂直向上夹角）
    spine_angle = frame_instance.get_spine_angle()
    
    # 绘制脊柱角度可视化（肩中心-髋中心-垂直向上的点）
    frame_instance.get_angle_and_draw('shldr_center', 'hip_center', 'vertical_up',
                                                     text_color='light_green',
                                                     line_color='light_blue', 
                                                     point_color='yellow')
    
    # 3D角度阈值可能需要调整，先用原有阈值测试
    # 脖子前倾判断和提示（鼻-肩-髋3D夹角）
    if neck_angle < 130:  # 3D角度可能更小，调整阈值
        frame_instance.show_feedback(text='严重低头！请抬起头', y=80, text_color='light_yellow', bg_color='red')
    elif neck_angle < 140:
        frame_instance.show_feedback(text='轻微低头，注意调整', y=80, text_color='black', bg_color='yellow')
    else:
        frame_instance.show_feedback(text='头部姿势良好', y=80, text_color='white', bg_color=(0, 128, 0))
        
    # 脊柱弯曲判断和提示（使用改进的脊柱角度）
    if spine_angle > 20:  # 真正的脊柱前倾角度阈值
        frame_instance.show_feedback(text='严重前倾！请挺直腰背', y=170, text_color='light_yellow', bg_color='red')
    elif spine_angle > 10:
        frame_instance.show_feedback(text='轻微前倾，注意坐姿', y=170, text_color='black', bg_color='yellow')
    else:
        frame_instance.show_feedback(text='脊柱姿势良好', y=170, text_color='white', bg_color=(0, 128, 0))
        
    # 显示角度数值
    frame_instance.draw_text(
        text='脖子前倾角度: ' + str(neck_angle) + '°',
        pos=(40, frame_height - 140),
        text_color=(255, 255, 255),
        font_scale=0.6,
        bg_color=(0, 100, 200),
    )
    
    frame_instance.draw_text(
        text='脊柱前倾角度: ' + str(spine_angle) + '°', 
        pos=(40, frame_height - 100),
        text_color=(255, 255, 255),
        font_scale=0.6,
        bg_color=(0, 100, 200),
    )
    
    # 坐姿总体评估
    if neck_angle >= 115 and spine_angle <= 10:
        frame_instance.draw_text(
            text='坐姿良好',
            pos=(40, frame_height - 50),
            text_color=(255, 255, 255),
            font_scale=0.8,
            bg_color=(0, 128, 0),
        )
    else:
        frame_instance.draw_text(
            text='需要调整坐姿',
            pos=(40, frame_height - 50),
            text_color=(255, 255, 255),
            font_scale=0.8,
            bg_color=(255, 140, 0),
        )