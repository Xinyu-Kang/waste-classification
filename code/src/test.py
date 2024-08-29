from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
import cv2
import numpy as np
import yaml

from segmentation import SegmentationModel 
from depth import DepthModel 
from strategy import SelectionStrategy
from draw import save_monitoring_image

app = Flask(__name__) 
CORS(app)
https_bp = Blueprint('https', __name__)
http_bp = Blueprint('http', __name__)

if __name__ == '__main__':

    # 直接读取配置文件
    with open('./config/yoloconfig.yaml', 'r') as file:
        config_seg = yaml.safe_load(file)
    with open('./config/depthconfig.yaml', 'r') as file:
        config_depth = yaml.safe_load(file)
    with open('./config/strategy.yaml', 'r') as file:
        strategy_config = yaml.safe_load(file)

    # 根据配置文件选择模型
    seg_model_name = config_seg['model']['selected']
    segmentation_model = SegmentationModel(seg_model_name)
    depth_model_name = config_depth['model']['selected']
    depth_model = DepthModel(depth_model_name)

    print("\n读取图片...")
    image = cv2.imread("../../small_test_data/photo_1.jpg")

    print("\n语义分割...")
    segmentation_results = segmentation_model.predict(image)
    label_names = segmentation_model.get_label_names()
    print(segmentation_results)
    
    print("\n深度预测...")
    depth_map = depth_model.predict(image)
    depth_array = np.asarray(depth_map)
    cv2.imwrite("../../small_test_data/depth_2.jpg", depth_map)

    print("\n选择物体...")
    strategy = SelectionStrategy(image, segmentation_results, depth_map, label_names)
    image_grab_point, label, points, all_candidates = strategy.select()
    print("grab point: ", image_grab_point)
    save_monitoring_image(image, all_candidates, image_grab_point, "photo_1.jpg", "../../monitoring")

    # # print(object)
    # image_grab_point = object["points"][0]
    # label = object["label"]
    # with app.app_context():
    #     result = jsonify({'point': image_grab_point, 'label': label})
    #     print("\n")
    #     print(result.data)
    # print("完成")