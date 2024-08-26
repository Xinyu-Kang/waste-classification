import cv2
import numpy as np
import yaml

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
        self.calculate_depth_score()
        self.calculate_area()
        best_candidate = None

        if self.candidates == []:
            print("no valid candidates")
            return None, None, None, None

        # 找出深度分数最大的候选区域
        max_depth_score = max(candidate['depth-score'] for candidate in self.candidates)
        max_depth_candidates = [
            candidate for candidate in self.candidates if candidate['depth-score'] == max_depth_score and candidate['is-selected']
        ]
        for candidate in max_depth_candidates:
            candidate['is-max-depth'] = True

        # 如果有多个深度分数相同的候选区域，则选择x坐标在指定范围内的那个
        for candidate in max_depth_candidates:
            grasp_point = self.calculate_grasp_point(candidate['points'])
            if self.is_within_x_range(grasp_point[0]):
                best_candidate = candidate
                break

        print(self.candidates)
        
        if best_candidate is None:
            return None, None, None, None

        # 返回选定区域的抓取点、标签及所有points
        return grasp_point, best_candidate['label'], best_candidate['points'], self.candidates 

    def calculate_depth_score(self):
        masks = self.segmentation_results[0].masks
        class_ids = self.segmentation_results[0].boxes.cls

        if masks is not None:
            for i, mask in enumerate(masks.xy):
                print("\nObject ", i)
                label = self.label_names[int(class_ids[i])]
                if label in self.filter_labels:
                    continue
                print("label: ", label)
                points = np.array(mask, dtype=np.int32)
                print("points: ", len(points))
                score = calculate_score(self.depth_map, points)
                print("score: ", score)
                candidate_info = {
                    "label": label,
                    "points": points,
                    "depth-score": score,
                    "is-max-candidate": False
                }
                candidate_info["is-selected"] = (score >= self.depth_threshold)
                self.candidates.append(candidate_info)

    def calculate_area(self):

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
        boundary_points = np.array(points, dtype=np.int32)
        mask = np.zeros(self.depth_map.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [boundary_points], color=255)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
        return max_loc
    
    def is_within_x_range(self, x):
        image_width = self.image.shape[1]
        left_limit = int(image_width * self.x_range[0])
        right_limit = int(image_width * (1 - self.x_range[1]))
        return left_limit <= x <= right_limit 


def calculate_score(depth_map, boundary_points):
    inside_dilation, outside_dilation = dilate_boundary(depth_map, boundary_points)
    num_points = len(boundary_points)
    num_greater = sum(
        compare_inside_outside(depth_map, point, inside_dilation, outside_dilation)
        for point in boundary_points
    )
    return num_greater / num_points

def dilate_boundary(depth_map, boundary_points):
    boundary_points = boundary_points.reshape((-1, 1, 2))
    mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
    cv2.polylines(mask, [boundary_points], isClosed=True, color=1, thickness=1)
    
    inside_mask = np.zeros_like(mask)
    cv2.fillPoly(inside_mask, [boundary_points], color=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    outside_dilation = cv2.subtract(dilated_mask, inside_mask)
    inside_dilation = cv2.subtract(dilated_mask, outside_dilation)

    return inside_dilation, outside_dilation

def compare_inside_outside(depth_map, point, inside_dilation, outside_dilation):
    # print(f"Processing point: {point}")
    box_mask = create_box(depth_map, point, (20, 20))
    box_inside_mask = cv2.bitwise_and(inside_dilation, box_mask)
    box_outside_mask = cv2.bitwise_and(outside_dilation, box_mask)

    inside_values = depth_map[box_inside_mask == 1]
    outside_values = depth_map[box_outside_mask == 1]
    inside_avg = np.mean(inside_values)
    outside_avg = np.mean(outside_values)

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
