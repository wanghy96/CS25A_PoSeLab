import cv2
import numpy as np
import math

from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line, calculate_angle_between_two_points

COLORS = {
    'black'         : (0, 0, 0),
    'blue'          : (0, 127, 255),
    'red'           : (255, 50, 50),
    'green'         : (0, 255, 127),
    'light_green'   : (100, 233, 127),
    'yellow'        : (255, 255, 0),
    'light_yellow'  : (255, 255, 230),
    'magenta'       : (255, 0, 255),
    'white'         : (255,255,255),
    'cyan'          : (0, 255, 255),
    'light_blue'    : (102, 204, 255)
}

LINE_TYPE = cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_SIMPLEX
OFFSET_THRESH = 35.0

class FrameInstance:
    def __init__(self, frame: np.array, pose):
        self.frame = frame
        self.pose = pose
        self.frame_height, self.frame_width, _ = frame.shape

        self.keypoints = pose.process(frame)

        self.coord = {
            'nose': None,
            'left_shldr': None,
            'left_elbow': None,
            'left_wrist': None,
            'left_hip': None,
            'left_knee': None,
            'left_ankle': None,
            'left_foot': None,
            'right_shldr': None,
            'right_elbow': None,
            'right_wrist': None,
            'right_hip': None,
            'right_knee': None,
            'right_ankle': None,
            'right_foot': None,
            'shldr_center': None,  # 双肩中心
            'hip_center': None,    # 双髋中心
        }
        self.angle = {}
        self.orientation = None

        if self.validate():
            ps_lm = self.keypoints.pose_landmarks
            self.coord['nose'] = get_landmark_features(ps_lm.landmark, 'nose', self.frame_width, self.frame_height)
            self.coord['left_shldr'], self.coord['left_elbow'], self.coord['left_wrist'], self.coord['left_hip'], self.coord['left_knee'], \
                self.coord['left_ankle'], self.coord['left_foot'] = get_landmark_features(ps_lm.landmark, 'left', self.frame_width, self.frame_height)
            self.coord['right_shldr'], self.coord['right_elbow'], self.coord['right_wrist'], self.coord['right_hip'], self.coord['right_knee'], \
                self.coord['right_ankle'], self.coord['right_foot'] = get_landmark_features(ps_lm.landmark, 'right', self.frame_width, self.frame_height)

            # 计算双肩中心和双髋中心
            self.coord['shldr_center'] = (self.coord['left_shldr'] + self.coord['right_shldr']) / 2
            self.coord['hip_center'] = (self.coord['left_hip'] + self.coord['right_hip']) / 2

    def validate(self):
        if self.keypoints.pose_landmarks:
            return True
        else:
            return False

    def get_frame(self):
        return self.frame

    def get_frame_width(self):
        return self.frame_width

    def get_frame_height(self):
        return self.frame_height

    def get_coord(self, feature):
        if self.validate():
            return self.coord[feature]
        else:
            return np.array([0, 0])

    def get_orientation(self):
        return self.orientation

    def get_angle(self, point1, point2, point3):
        angle, coord1, coord2, coord3  = self.__get_angle__(point1, point2, point3)
        return int(angle)

    def get_angle_and_draw(self, point1, point2, point3, text_color='light_green', line_color='light_blue', point_color='yellow', ellipse_color='white', dotted_line_color='blue'):
        angle, coord1, coord2, coord3 = self.__get_angle__(point1, point2, point3)

        # 转换为2D坐标用于绘制
        coord1_2d = self.__convert_3d_to_2d__(coord1)
        coord2_2d = self.__convert_3d_to_2d__(coord2)
        coord3_2d = self.__convert_3d_to_2d__(coord3)

        # 直接在2D坐标上绘制角度，避免坐标系转换问题
        # 使用更简单和直观的角度绘制方法
        self.__draw_angle_arc__(coord1_2d, coord2_2d, coord3_2d, ellipse_color)

        # draw lines between points
        self.line(point1, point2, line_color, 4)
        if point3 == 'vertical' or point3 == 'horizontal' or point3 == 'nvertical' or point3 == 'nhorizontal' or point3 == 'vertical_up':
            # draw vertical or horizontal line (使用2D坐标)
            draw_dotted_line(self.frame, coord2_2d, start=coord2_2d[1] - 50, end=coord2_2d[1] + 20,
                             line_color=self.__get_color__(dotted_line_color))
        else:
            self.line(point2, point3, line_color, 4)

        # draw point cicle
        self.circle(point1, radius=7, color=point_color)
        self.circle(point2, radius=7, color=point_color)
        if point3 in self.coord:
            self.circle(point3, radius=7, color=point_color)

        # show angle value (使用2D坐标，优化位置避免遮挡)
        text_offset_x = 25
        text_offset_y = -10
        cv2.putText(self.frame, str(int(angle)), 
                   (coord2_2d[0] + text_offset_x, coord2_2d[1] + text_offset_y), 
                   FONT, 0.6,
                   self.__get_color__(text_color), 2, lineType=LINE_TYPE)

        return int(angle)
    
    def __draw_angle_arc__(self, p1_2d, p2_2d, p3_2d, color):
        """
        在2D坐标上绘制角度弧线，使用更直观的方法
        """
        try:
            # 计算两个向量
            v1 = p1_2d - p2_2d  # 向量1: p2 -> p1
            v2 = p3_2d - p2_2d  # 向量2: p2 -> p3
            
            # 计算向量的角度（相对于X轴正方向）
            angle1 = math.degrees(math.atan2(v1[1], v1[0]))
            angle2 = math.degrees(math.atan2(v2[1], v2[0]))
            
            # 标准化角度到 0-360 范围
            angle1 = (angle1 + 360) % 360
            angle2 = (angle2 + 360) % 360
            
            # 计算角度差
            angle_diff = (angle2 - angle1 + 360) % 360
            
            # 如果角度大于180度，绘制补角
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
                start_angle = angle2
                end_angle = angle1
            else:
                start_angle = angle1
                end_angle = angle2
            
            # 绘制椭圆弧线
            cv2.ellipse(self.frame, 
                       p2_2d, (20, 20), 
                       angle=0, 
                       startAngle=start_angle, 
                       endAngle=end_angle,
                       color=self.__get_color__(color), 
                       thickness=3, 
                       lineType=LINE_TYPE)
                       
        except Exception as e:
            # 如果绘制失败，至少画出连线
            pass

    def circle(self, *args, radius=7, color='yellow'):
        for arg in args:
            coord_2d = self.__convert_3d_to_2d__(self.coord[arg])
            cv2.circle(self.frame, coord_2d, radius, self.__get_color__(color), -1)
            # cv2.circle(self.frame, self.coord[center], radius, self.__get_color__(color), -1, LINE_TYPE)

    def line(self, pt1, pt2, color='light_blue', thickness=4):
        coord1_2d = self.__convert_3d_to_2d__(self.coord[pt1])
        coord2_2d = self.__convert_3d_to_2d__(self.coord[pt2])
        cv2.line(self.frame, coord1_2d, coord2_2d, self.__get_color__(color), thickness, LINE_TYPE)

    def draw_text(self, text, width=8, font=FONT, pos=(0, 0), font_scale=1.0, font_thickness=2, text_color=(0, 255, 0)
                  , bg_color=(0, 0, 0)):
        self.frame = draw_text(self.frame, text, width, font, pos, font_scale, font_thickness, text_color, bg_color)

    def put_text(self, text, pos, font_scale, color, thickness, line_type=LINE_TYPE):
        cv2.putText(self.frame, text=text, org=pos, fontFace=FONT, fontScale=font_scale,
                    color=self.__get_color__(color), thickness=thickness, lineType=line_type)

    def show_feedback(self, text, y, text_color, bg_color):
        self.draw_text(
            text,
            pos=(30, y),
            text_color=self.__get_color__(text_color),
            font_scale=0.6,
            bg_color=self.__get_color__(bg_color)
        )

        return self.frame

    def __get_angle__(self, point1, point2, point3):
        key = point1 + '#' + point2 + '#' + point3
        if key not in self.angle:
            coord1 = self.get_coord(point1)
            coord2 = self.get_coord(point2)
            coord3 = None
            if point3 in self.coord:
                coord3 = self.get_coord(point3)
            else:
                if point3 == 'vertical':
                    coord3 = self.__get_vertical_coord__(point2)
                if point3 == 'horizontal':
                    coord3 = self.__get_horizontal_coord__(point2)
                if point3 == 'nvertical':
                    coord3 = self.__get_nvertical_coord__(point2)
                if point3 == 'nhorizontal':
                    coord3 = self.__get_nhorizontal_coord__(point2)
                if point3 == 'vertical_up':
                    coord3 = self.__get_vertical_up_coord__(point2)

            if coord3 is None:
                return 0, np.array([0, 0]), np.array([0, 0]), np.array([0, 0])

            angle = find_angle(coord1, coord3, coord2)
            self.angle[key] = {
                'angle': angle,
                'coord1': coord1,
                'coord2': coord2,
                'coord3': coord3,
            }

        return self.angle[key]['angle'], self.angle[key]['coord1'], self.angle[key]['coord2'], self.angle[key]['coord3']

    def __get_color__(self, color):
        if isinstance(color, str):
            return COLORS[color]
        else:
            return color

    def __get_vertical_coord__(self, feature):
        coord = self.get_coord(feature)
        return np.array([coord[0], coord[1] + 0.3, coord[2]])  # 向下增加0.3的偏移（Y向下为正）

    def __get_horizontal_coord__(self, feature):
        coord = self.get_coord(feature)
        return np.array([coord[0] - 0.3, coord[1], coord[2]])  # 向左减少0.3的偏移

    def __get_nvertical_coord__(self, feature):
        coord = self.get_coord(feature)
        return np.array([coord[0], coord[1] - 0.3, coord[2]])  # 向上减少0.3的偏移（Y向上为负）

    def __get_nhorizontal_coord__(self, feature):
        coord = self.get_coord(feature)
        return np.array([coord[0] + 0.3, coord[1], coord[2]])  # 向右增加0.3的偏移
    
    def __get_vertical_up_coord__(self, feature):
        """获取垂直向上的点"""
        coord = self.get_coord(feature)
        return np.array([coord[0], coord[1] - 0.3, coord[2]])  # 垂直向上0.3

    def __convert_coord__(self, coord, origin_coord):
        return np.array([coord[0] - origin_coord[0], coord[1] - origin_coord[1], coord[2] - origin_coord[2]])

    def __convert_3d_to_2d__(self, coord_3d):
        """将3D归一化坐标转换为2D屏幕坐标用于绘制"""
        x_2d = int(coord_3d[0] * self.frame_width)
        y_2d = int(coord_3d[1] * self.frame_height)
        return np.array([x_2d, y_2d])
    
    def get_spine_angle(self):
        """
        计算脊柱弯曲角度：肩中心-髋中心-垂直向上的夹角
        返回身体前倾的程度（相对于垂直向上的角度）
        """
        if self.coord['shldr_center'] is None or self.coord['hip_center'] is None:
            return 0
        
        shldr_center = self.coord['shldr_center']
        hip_center = self.coord['hip_center']
        
        # 计算肩到髋的向量（从上到下的方向）
        spine_vector = hip_center - shldr_center
        
        # 垂直向上向量（Y轴负方向，因为归一化坐标中Y向下，向上为负值）
        vertical_vector = np.array([0, -0.1, 0])  # 向上偏移
        
        # 计算脊柱向量与垂直向上向量的夹角
        # 使用髋部作为参考点，计算肩部-髋部-垂直向上点的夹角
        vertical_up_point = hip_center + vertical_vector  # 髋部垂直向上的点
        angle = find_angle(shldr_center, vertical_up_point, hip_center)
        
        return angle
    
    def draw_spine_line(self):
        """绘制脊柱线（肩中心到髋中心）"""
        if self.coord['shldr_center'] is not None and self.coord['hip_center'] is not None:
            shldr_2d = self.__convert_3d_to_2d__(self.coord['shldr_center'])
            hip_2d = self.__convert_3d_to_2d__(self.coord['hip_center'])
            cv2.line(self.frame, shldr_2d, hip_2d, self.__get_color__('light_blue'), 4, LINE_TYPE)
            
            # 在肩中心和髋中心画点
            cv2.circle(self.frame, shldr_2d, 7, self.__get_color__('yellow'), -1, LINE_TYPE)
            cv2.circle(self.frame, hip_2d, 7, self.__get_color__('yellow'), -1, LINE_TYPE)