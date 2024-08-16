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
    config = yaml.safe_load(file)

# 根据配置文件选择模型
model_name = config['model']['selected']
segmentation_model = SegmentationModel(model_name)


# depth_model = DepthModel('depth-anything')
# strategy = SelectionStrategy()


@app.route('/process_image', methods=['POST'])
def process_image():

    image = get_image(request.files['image'])

    segmentation_results = segmentation_model.predict(image)

    # depth_results = depth_model.predict(image, segmentation_results)

    # image_point, label = strategy.select(segmentation_results, depth_results)

    
    return jsonify({'grab_point': 1, 'label': "haha"})


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