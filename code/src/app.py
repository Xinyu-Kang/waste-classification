from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
import cv2
import numpy as np
import yaml

from segmentation import SegmentationModel 
from depth import DepthModel 
from strategy import SelectionStrategy

app = Flask(__name__) 
CORS(app)
https_bp = Blueprint('https', __name__)
http_bp = Blueprint('http', __name__)

# 直接读取配置文件
with open('./config/yoloconfig.yaml', 'r') as file:
    config_seg = yaml.safe_load(file)
with open('./config/depthconfig.yaml', 'r') as file:
    config_depth = yaml.safe_load(file)

# 根据配置文件选择模型
seg_model_name = config_seg['model']['selected']
segmentation_model = SegmentationModel(seg_model_name)
depth_model_name = config_depth['model']['selected']
depth_model = DepthModel(depth_model_name)


@app.route('/process_image', methods=['POST'])
def process_image():
    """
    从request拿到现场图片,进行语义分割和深度预测,并使用策略选择抓取物体。
    :return:物体抓取点在图片上的坐标及物体的类别
    """
    image = get_image(request.files['image'])

    segmentation_results = segmentation_model.predict(image)
    label_names = segmentation_model.get_label_names()

    depth_map = depth_model.predict(image)

    strategy = SelectionStrategy(image, segmentation_results, depth_map, label_names)
    
    image_grab_point, label = strategy.select()
    return jsonify({'point': image_grab_point.tolist(), 'label': label})


def get_image(data):  
    # 将文件数据读取到一个字节流中
    image_np = np.frombuffer(data.read(), np.uint8)
    
    # 解码图像
    image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
    
    return image



if __name__ == '__main__':
    app.register_blueprint(http_bp)
    app.register_blueprint(https_bp)
    app.run(host='0.0.0.0', port=5001)