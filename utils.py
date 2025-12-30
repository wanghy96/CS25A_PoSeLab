import math

import cv2
import mediapipe as mp
import numpy as np

def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):

    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width

    # draw filled rectangles
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)


    # draw filled ellipses
    cv2.ellipse(img, (x1 + w, y1 + w), (w, w),
                angle = 0, startAngle = -90, endAngle = -180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y1 + w), (w, w),
                angle = 0, startAngle = 0, endAngle = -90, color = box_color, thickness = -1)

    cv2.ellipse(img, (x1 + w, y2 - w), (w, w),
                angle = 0, startAngle = 90, endAngle = 180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y2 - w), (w, w),
                angle = 0, startAngle = 0, endAngle = 90, color = box_color, thickness = -1)

    return img




def draw_dotted_line(frame, lm_coord, start, end, line_color):
    pix_step = 0

    for i in range(start, end+1, 8):
        cv2.circle(frame, (lm_coord[0], i+pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)

    return frame

from PIL import Image, ImageDraw, ImageFont
 
def draw_zh(
    img,
    msg,
    pos, 
    text_color,
    ):
    # cv2读取图
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
    pil_img = Image.fromarray(cv2img)
    
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pil_img) # 图片上打印
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    text_color_bgr = (text_color[2], text_color[1], text_color[0])  # 将RGB转换为BGR
    draw.text(pos, msg, text_color, font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    
    # PIL图片转cv2 图片
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

def draw_text(
    img,
    text,
    width = 8,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    pos=(0, 0),
    font_scale=1,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
    box_offset=(20, 10),
):
    msg = text
    offset = box_offset
    x, y = pos

    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    left, top, right, bottom = font.getbbox(msg)
    text_w = right - left
    text_h = bottom - top

    #text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    #text_w, text_h = text_size
    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(m + n - o for m, n, o in zip((x + text_w, y + text_h), offset, (25, 0)))

    img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)
    
    img = draw_zh(img,
        msg,
        (int(rec_start[0] + 10), int(y + text_h + font_scale) - 19), 
        text_color,
        )
    
    return img




import math

def find_angle(p1, p2, ref_pt = np.array([0,0,0])):
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref,p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
    degree = int(180 / np.pi) * theta

    if not math.isnan(degree):
        return int(degree)






def get_landmark_array(pose_landmark, key, frame_width, frame_height):

    denorm_x = pose_landmark[key].x
    denorm_y = pose_landmark[key].y
    denorm_z = pose_landmark[key].z

    return np.array([denorm_x, denorm_y, denorm_z])


dict_features = {
    'nose': 0,
    'left': {
        'shoulder': 11,
        'elbow': 13,
        'wrist': 15,
        'hip': 23,
        'knee': 25,
        'ankle': 27,
        'foot': 31
    },
    'right': {
        'shoulder': 12,
        'elbow': 14,
        'wrist': 16,
        'hip': 24,
        'knee': 26,
        'ankle': 28,
        'foot': 32
    }
}
def get_landmark_features(kp_results, feature, frame_width, frame_height):

    if feature == 'nose':
        return get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)

    elif feature == 'left' or 'right':
        shldr_coord = get_landmark_array(kp_results, dict_features[feature]['shoulder'], frame_width, frame_height)
        elbow_coord   = get_landmark_array(kp_results, dict_features[feature]['elbow'], frame_width, frame_height)
        wrist_coord   = get_landmark_array(kp_results, dict_features[feature]['wrist'], frame_width, frame_height)
        hip_coord   = get_landmark_array(kp_results, dict_features[feature]['hip'], frame_width, frame_height)
        knee_coord   = get_landmark_array(kp_results, dict_features[feature]['knee'], frame_width, frame_height)
        ankle_coord   = get_landmark_array(kp_results, dict_features[feature]['ankle'], frame_width, frame_height)
        foot_coord   = get_landmark_array(kp_results, dict_features[feature]['foot'], frame_width, frame_height)

        return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord
    
    else:
       raise ValueError("feature needs to be either 'nose', 'left' or 'right")


def get_mediapipe_pose(
                        static_image_mode = False, 
                        model_complexity = 1,
                        smooth_landmarks = True,
                        min_detection_confidence = 0.5,
                        min_tracking_confidence = 0.5

                      ):
    pose = mp.solutions.pose.Pose(
                                    static_image_mode = static_image_mode,
                                    model_complexity = model_complexity,
                                    smooth_landmarks = smooth_landmarks,
                                    min_detection_confidence = min_detection_confidence,
                                    min_tracking_confidence = min_tracking_confidence
                                 )
    return pose

def calculate_angle_between_two_points(point1, point2):
    """
    Calculate the angle between two points
    """
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    return math.degrees(math.atan2(y_diff, x_diff))