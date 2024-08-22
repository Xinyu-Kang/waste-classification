import cv2
import numpy as np
import os

def save_monitoring_image(image, segmentation_results, strategy_candidates, filename, monitoring_dir):
    """
    异步保存监控图片的函数
    :param image: 原始图像
    :param segmentation_results: YOLO分割结果
    :param strategy_candidates: 筛选后的候选物体，包含深度分数和标签
    :param filename: 保存图像的文件名
    """
    try:
        # 在图像上绘制Bounding Box和标签
        image_with_bboxes = draw_bboxes(image, strategy_candidates)
        # 保存监控图像到本地文件夹
        monitoring_filename = os.path.join(monitoring_dir, filename)
        cv2.imwrite(monitoring_filename, image_with_bboxes)
        
        print(f"Image saved to {monitoring_filename}")
    except Exception as e:
        print(f"Failed to save monitoring image: {e}")

def draw_bboxes(image, filtered_shapes):
    """
    在图像上绘制经过筛选的结果
    :param image: 原始图像
    :param filtered_shapes: 筛选后的候选物体，包含深度分数和标签
    :return: 带有Bounding Box和标签的图像
    """
    # 确保图像为三通道并且为uint8类型
    if image.shape[-1] == 4:  # 如果图像是四通道
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = np.array(image, dtype=np.uint8)
    
    for shape in filtered_shapes:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32).reshape((-1, 1, 2))
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        
        # 绘制多边形边界
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(points)
        
        # 绘制标签和深度分数
        cv2.putText(image, f'{label}: depth_score: {shape["depth-score"]:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    return image
