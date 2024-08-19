import cv2
import numpy as np
import yaml
from matplotlib.pyplot import imshow
from PIL import Image

from segmentation import SegmentationModel 
from depth import DepthModel 
from strategy import SelectionStrategy

if __name__ == '__main__':
    print("读取配置文件...")
    with open('./config/yoloconfig.yaml', 'r') as file:
        config_seg = yaml.safe_load(file)
    with open('./config/depthconfig.yaml', 'r') as file:
        config_depth = yaml.safe_load(file)

    print("选择模型...")
    seg_model_name = config_seg['model']['selected']
    segmentation_model = SegmentationModel(seg_model_name)
    depth_model_name = config_depth['model']['selected']
    depth_model = DepthModel(depth_model_name)

    print("读取图片...")
    image = cv2.imread("../../small_test_data/photo_1.jpg")

    print("语义分割...")
    segmentation_results = segmentation_model.predict(image)
    label_names = segmentation_model.get_label_names()

    print("深度预测...")
    depth_map = depth_model.predict(image)
    depth_array = np.asarray(depth_map)
    imshow(depth_array)