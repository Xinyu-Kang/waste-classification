import cv2
import numpy as np
import yaml
import random

class SelectionStrategy:

    def __init__(self, image, segmentation_results, depth_map, label_names) -> None:
        self.image = image
        self.segmentation_results = segmentation_results
        self.depth_map = depth_map
        self.label_names = label_names
        self.depth_threshold = 0
        self.area_threshold = 0
        self.candidates = []

        # 加载策略配置
        self.load_strategy_config()
    
    def load_strategy_config(self):
        with open('./config/strategy.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # 从配置中获取深度阈值、过滤标签和x轴范围比例
        self.depth_threshold = config.get('depth_threshold', 0)  # 如果没有定义，默认值为0
        self.filter_labels = config.get('filter_labels', [])  # 如果没有定义，默认值为空列表
        self.x_range = config.get('x_range', [0, 0])  # 默认允许整个图像范围内的x坐标

    def select(self):
        # 计算深度得分和面积
        self.calculate_depth_score()
        self.calculate_area()

        # 如果没有候选物体，返回None
        if not self.candidates:
            print("No valid candidates")
            return None, None, None, None

        # 从所有满足条件的物体中，随机选择一个
        valid_candidates = [candidate for candidate in self.candidates if candidate['is-selected']]

        if not valid_candidates:
            print("No valid candidates after filtering")
            return None, None, None, None

        # 随机选择一个候选物体
        best_candidate = random.choice(valid_candidates)
        best_candidate['is-best'] = True
        print("best_candidate id: ", best_candidate['id'])

        # 计算抓取点
        grasp_point = self.calculate_grasp_point(best_candidate['points'])
        
        # 返回选定区域的抓取点、标签及所有points
        return grasp_point, best_candidate['label'], best_candidate['points'], self.candidates


    def calculate_depth_score(self):
        # print("\nCalculating depth score...\n")
        masks = self.segmentation_results[0].masks
        class_ids = self.segmentation_results[0].boxes.cls

        if masks is not None:
            for i, mask in enumerate(masks.xy):
                # print("\nObject ", i)
                label = self.label_names[int(class_ids[i])]
                if label in self.filter_labels:
                    continue
                # print("label: ", label)
                points = np.array(mask, dtype=np.int32)
                # print("points: ", points.shape)
                if points.shape[0] == 0:
                    continue
                score = calculate_score(self.depth_map, points)
                # print("score: ", score)
                candidate_info = {
                    "id": i,
                    "label": label,
                    "points": points,
                    "depth-score": score,
                    "is-max-depth": False,
                    "is-best": False
                }
                candidate_info["is-selected"] = (score >= self.depth_threshold)
                self.candidates.append(candidate_info)

    def calculate_area(self):
        # print("\nCalculating area...\n")
        if self.candidates is None:
            raise ValueError("No candidates found. Please run the selection process first.")

        for candidate in self.candidates:
            # 获取候选区域的边界点
            boundary_points = np.array(candidate['points'], dtype=np.int32)
            
            # 计算面积
            area = cv2.contourArea(boundary_points)
            
            # 只保留面积大于阈值的物体
            candidate["is-selected"] = candidate["is-selected"] and (area > self.area_threshold)

    def calculate_grasp_point(self, points):
        # print("\nCalculating grasp point...\n")
        boundary_points = np.array(points, dtype=np.int32)
        mask = np.zeros(self.depth_map.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [boundary_points], color=255)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
        return max_loc
    
    def is_within_x_range(self, x):
        # print("\nCalculating x range...\n")
        image_width = self.image.shape[1]
        left_limit = int(image_width * self.x_range[0])
        right_limit = int(image_width * (1 - self.x_range[1]))
        return left_limit <= x <= right_limit 


def calculate_score(depth_map, boundary_points):
    # print("   calculate_score()")
    inside_dilation, outside_dilation = dilate_boundary(depth_map, boundary_points)

    num_points = len(boundary_points)
    num_greater = sum(
        compare_inside_outside(depth_map, point, inside_dilation, outside_dilation)
        for point in boundary_points
    )
    # print(f"   {num_points}/{num_greater}\n")
    return num_greater / num_points

def dilate_boundary(depth_map, boundary_points):
    # print("    dilate_boundary()")
    boundary_points = boundary_points.reshape((-1, 1, 2))
    mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
    cv2.polylines(mask, [boundary_points], isClosed=True, color=1, thickness=1)
    # print("     calculated mask")
    inside_mask = np.zeros_like(mask, dtype=np.uint8)
    # print("     inside_mask:", inside_mask.shape)
    # print("     boundary_points: ", boundary_points.shape)
    cv2.fillPoly(inside_mask, [boundary_points], color=1)
    # print("     calculated inside_mask")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    # print("     calculated outside_mask")
    outside_dilation = cv2.subtract(dilated_mask, inside_mask)
    inside_dilation = cv2.subtract(dilated_mask, outside_dilation)

    return inside_dilation, outside_dilation

def compare_inside_outside(depth_map, point, inside_dilation, outside_dilation):
    # print(f"    compare_inside_outside()")
    box_mask = create_box(depth_map, point, (20, 20))
    box_inside_mask = cv2.bitwise_and(inside_dilation, box_mask)
    box_outside_mask = cv2.bitwise_and(outside_dilation, box_mask)

    inside_values = depth_map[box_inside_mask == 1]
    outside_values = depth_map[box_outside_mask == 1]
    inside_avg = np.mean(inside_values)
    outside_avg = np.mean(outside_values)
    # print("    ", inside_avg >= outside_avg)
    return int(inside_avg >= outside_avg)

def create_box(image, center_point, box_size):
    # 确保center_point是一个长度为2的数组或元组
    if len(center_point) != 2:
        raise ValueError(f"Expected center_point to be a tuple or list of length 2, got {center_point}")
    x_center, y_center = center_point
    box_width, box_height = box_size
    
    x_start = max(x_center - box_width // 2, 0)
    y_start = max(y_center - box_height // 2, 0)
    x_end = min(x_center + box_width // 2, image.shape[1])
    y_end = min(y_center + box_width // 2, image.shape[0])

    box = np.zeros(image.shape[:2], dtype=np.uint8)
    box[y_start:y_end, x_start:x_end] = 1
    return box
