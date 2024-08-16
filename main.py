import os
import cv2
import torch
import numpy as np
import yaml
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from depth_module.depth_score_module import calculate_depth_score  # 从depth_score.py导入函数

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载配置文件
    config = load_config('./config/config.yaml')
    depth_score_threshold = config.get('depth_score_threshold', 0.9)
    
    # 加载 YOLO 和 DepthAnythingV2 模型
    yolo_model = YOLO('./yolo_models/best.pt').to(device)
    model_names = yolo_model.names  # 获取模型的类别名称
    encoder = 'vits'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything_model = DepthAnythingV2(**model_configs[encoder])
    depth_anything_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location=device))
    depth_anything_model = depth_anything_model.to(device).eval()

    # 加载图片并进行预测
    image_path = './image2.png'
    image = cv2.imread(image_path)

    # YOLO识别
    yolo_results = yolo_model(image)

    # 生成深度图
    depth_map = depth_anything_model.infer_image(image, input_size=518)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map = depth_map.astype(np.uint8)

    # 计算深度分数
    filtered_shapes = calculate_depth_score(image, yolo_results, depth_map, model_names, depth_score_threshold)

    # 打印bbox和深度分数
    for shape in filtered_shapes:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)

        # 绘制边界框和标签
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制边框
        cv2.putText(image, f'{label}: depth score {shape["depth-score"]:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 绘制标签和深度分数
        
    
        
    # 创建results文件夹
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存处理后的图片，保持图片名一致
    image_filename = os.path.basename(image_path)
    output_image_path = os.path.join(results_dir, image_filename)
    cv2.imwrite(output_image_path, image)

    # print(f"Processed image saved to {output_image_path}.")
