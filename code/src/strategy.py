import cv2
import numpy as np

class SelectionStrategy:

    def __init__(self, image, segmentation_results, depth_map, label_names) -> None:
        self.image = image
        self.segmentation_results = segmentation_results
        self.depth_results = depth_map
        self.label_names = label_names
        self.threshold = 0.95
        self.candidates = None

    def select(self):
        self.calculate_depth_score()
        self.sum_boarder_distance()
        self.min_boarder_distance()
        self.calculate_area()
        image_grab_point = None
        label = None
        return image_grab_point, label

    def calculate_depth_score(self):
        filtered_shapes = []
        masks = self.segmentation_results[0].masks  # 获取分割结果
        class_ids = self.segmentation_results[0].boxes.cls  # 获取类别ID

        if masks is not None:
            for i, mask in enumerate(masks.data):  # 直接使用掩码数据
                label = self.label_names[int(class_ids[i])]

                # 将 Tensor 转换为 NumPy 数组并转换类型
                mask = mask.cpu().numpy().astype(np.uint8)
                
                # 提取掩码的边界
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                boundary_points = contours[0].reshape(-1, 2)  # 获取边界上的所有点

                # 计算深度得分
                score = calculate_score(self.depth_map, boundary_points)
                if score > self.threshold:
                    filtered_shapes.append({
                        "label": label,
                        "points": boundary_points.tolist(),
                        "depth-score": score
                    })
        self.candidates = filtered_shapes

    def sum_boarder_distance(self):
        pass

    def min_boarder_distance(self):
        pass

    def calculate_area(self):
        pass

def calculate_score(image, boundary_points):
    inside_dilation, outside_dilation = dilate_boundary(image, boundary_points)
    num_points = len(boundary_points)
    num_greater = sum(
        compare_inside_outside(image, point, inside_dilation, outside_dilation)
        for point in boundary_points
    )
    return num_greater / num_points

def dilate_boundary(image, boundary_points):
    boundary_points = boundary_points.reshape((-1, 1, 2))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.polylines(mask, [boundary_points], isClosed=True, color=1, thickness=1)
    
    inside_mask = np.zeros_like(mask)
    cv2.fillPoly(inside_mask, [boundary_points], color=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    outside_dilation = cv2.subtract(dilated_mask, inside_mask)
    inside_dilation = cv2.subtract(dilated_mask, outside_dilation)

    return inside_dilation, outside_dilation

def compare_inside_outside(image, point, inside_dilation, outside_dilation):
    print(f"Processing point: {point}")
    box_mask = create_box(image, point, (20, 20))
    box_inside_mask = cv2.bitwise_and(inside_dilation, box_mask)
    box_outside_mask = cv2.bitwise_and(outside_dilation, box_mask)

    inside_values = image[box_inside_mask == 1]
    outside_values = image[box_outside_mask == 1]
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