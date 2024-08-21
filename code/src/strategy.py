import cv2
import numpy as np

class SelectionStrategy:

    def __init__(self, image, segmentation_results, depth_map, label_names) -> None:
        self.image = image
        self.segmentation_results = segmentation_results
        self.depth_map = depth_map
        self.label_names = label_names
        self.depth_threshold = 0
        self.area_threshold = 0
        self.candidates = None

    def select(self):
        self.calculate_depth_score()
        self.calculate_boarder_distance()
        # self.calculate_area()
        best_candidate = None
        best_depth_score = -1
        valid_candidates = self.candidates
        

        # 设置面积的最小阈值
        area_threshold = 0  # 你可以根据实际情况调整这个值
        
        # 过滤出面积大于阈值的候选区域
        # valid_candidates = [
        #     candidate for candidate in self.candidates if candidate['area'] > area_threshold
        # ]
        
        if not valid_candidates:
            print("no valied canditates")
            return None, None, None
        
        # 找出深度分数最大的候选区域
        max_depth_score = max(candidate['depth-score'] for candidate in valid_candidates)
        max_depth_candidates = [
            candidate for candidate in valid_candidates if candidate['depth-score'] == max_depth_score
        ]

        # 如果有多个深度分数相同的候选区域，则选择 x 坐标最大的那个
        if len(max_depth_candidates) > 1:
            max_depth_candidates.sort(key=lambda x: max(np.array(x['points'])[:, 0]), reverse=True)

        best_candidate = max_depth_candidates[0]

        # 计算抓取点
        grasp_point = self.calculate_grasp_point(best_candidate['points'])

        # 返回选定区域的抓取点、标签及所有points
        label = best_candidate['label']

        return grasp_point, label, best_candidate['points']
    


    def calculate_grasp_point(self, points):
        # 将points转换为NumPy数组
        boundary_points = np.array(points, dtype=np.int32)
        
        # 创建一个二值掩码图像
        mask = np.zeros(self.depth_map.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [boundary_points], color=255)
        
        # 计算距离变换
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
        
        return max_loc

    def calculate_depth_score(self):
        filtered_shapes = []
        masks = self.segmentation_results[0].masks  # 获取分割结果
        class_ids = self.segmentation_results[0].boxes.cls  # 获取类别ID
        
        if masks is not None:
            for i, mask in enumerate(masks.xy):
                label = self.label_names[int(class_ids[i])]  # 从模型中获取类别名称
                points = np.array(mask, dtype=np.int32)  # 获取多边形的像素点坐标

                # resampled_points = uniform_resample(points, 2)
                
                # resampled_points = np.array(resampled_points, dtype=np.int32)

                # 计算深度得分
                score = calculate_score(self.depth_map, points)
                if score > self.depth_threshold:
                    filtered_shapes.append({
                        "label": label,
                        "points": points.tolist(),
                        "depth-score": score
                    })

        self.candidates = filtered_shapes

    # def calculate_depth_score(self):
    #     filtered_shapes = []
    #     masks = self.segmentation_results[0].masks  # 获取分割结果
    #     class_ids = self.segmentation_results[0].boxes.cls  # 获取类别ID

    #     if masks is not None:
    #         for i, mask in enumerate(masks.data):  # 直接使用掩码数据
    #             label = self.label_names[int(class_ids[i])]

    #             # 将 Tensor 转换为 NumPy 数组并转换类型
    #             mask = mask.cpu().numpy().astype(np.uint8)
                
    #             # 提取掩码的边界
    #             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #             boundary_points = contours[0].reshape(-1, 2)  # 获取边界上的所有点

    #             # 计算深度得分
    #             score = calculate_score(self.depth_map, boundary_points)
    #             if score > self.depth_threshold:
    #                 filtered_shapes.append({
    #                     "label": label,
    #                     "points": boundary_points.tolist(),
    #                     "depth-score": score
    #                 })
    #     self.candidates = filtered_shapes

    def calculate_boarder_distance(self):
        for i in range(len(self.candidates)):
            points = np.array(self.candidates[i]['points'])
            top, left = np.min(points, axis=0)
            bottom, right = np.max(points, axis=0)
            total_distance = left + (640 - right) + top + (480 - bottom)
            min_distance = min(left, 640 - right, top, 480 - bottom)
            self.candidates[i]['total-distance'] = total_distance
            self.candidates[i]['min-distance'] = min_distance

    def calculate_area(self):

        if self.candidates is None:
            raise ValueError("No candidates found. Please run the selection process first.")

        filtered_candidates = []
        for candidate in self.candidates:
            # 获取候选区域的边界点
            boundary_points = np.array(candidate['points'], dtype=np.int32)
            
            # 计算面积
            area = cv2.contourArea(boundary_points)
            
            # 只保留面积大于阈值的物体
            if area > self.area_threshold:
                candidate['area'] = area
                filtered_candidates.append(candidate)

        # 更新 self.candidates，只保留符合面积条件的对象
        self.candidates = filtered_candidates

def calculate_score(depth_map, boundary_points):
    inside_dilation, outside_dilation = dilate_boundary(depth_map, boundary_points)
    num_points = len(boundary_points)
    num_greater = sum(
        compare_inside_outside(depth_map, point, inside_dilation, outside_dilation)
        for point in boundary_points
    )
    print(num_greater / num_points)
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
    print(f"Processing point: {point}")
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
