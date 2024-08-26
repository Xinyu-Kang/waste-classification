import cv2
import numpy as np
import os

def save_monitoring_image(image, candidates, grab_point, filename, monitoring_dir):
    """
    异步保存监控图片的函数
    :param image: 原始图像
    :param segmentation_results: YOLO分割结果
    :param grab_point: 策略选出的抓取点
    :param filename: 保存图像的文件名
    """
    try:
        # 在图像上绘制Bounding Box和抓取点
        image_with_bboxes = draw_bboxes(image, candidates, grab_point)
        
        # 保存监控图像到本地文件夹
        monitoring_filename = os.path.join(monitoring_dir, filename)
        cv2.imwrite(monitoring_filename, image_with_bboxes)
        
        print(f"Image saved to {monitoring_filename}")
    except Exception as e:
        print(f"Failed to save monitoring image: {e}")

def draw_bboxes(image, candidates, grab_point):
    """
    在图像上绘制经过筛选的结果，并突出显示策略选出的抓取点。
    :param image: 原始图像
    :param segmentation_results: YOLO分割结果
    :param grab_point: 策略选出的抓取点坐标 (x, y)
    :return: 带有Bounding Box和标签的图像
    """
    # 确保图像为三通道并且为uint8类型
    if image.shape[-1] == 4:  # 如果图像是四通道
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = np.array(image, dtype=np.uint8)

    # 绘制YOLO的分割结果
    for c in candidates:
        label = c['label']
        points = c['points'].reshape((-1, 1, 2))  # 使用 xy 数据进行绘制
        depth_score = c['depth-score']
        is_selected = c['is-selected']
        is_max_depth = c['is-max-depth']
        is_best = c['is-best']

        color = (0, 0, 255) if is_best else (0, 255, 0) if is_selected else (255, 0, 0)
        # color = (255, 0, 0) if is_best else (0, 255, 0)

        # 绘制多边形边界
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(points)
        
        # 绘制标签和深度分数
        cv2.putText(image, f'{label}: {depth_score:.2f}', (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # 绘制抓取点
    if grab_point is not None:
        cv2.circle(image, grab_point, 5, (0, 0, 255), -1)  # 红色圆点表示抓取点

    return image
