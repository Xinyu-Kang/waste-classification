from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
import cv2
import numpy as np

from segmentation import SegmentationModel 
from depth import DepthModel 
from strategy import SelectionStrategy

app = Flask(__name__) 
CORS(app)
https_bp = Blueprint('https', __name__)
http_bp = Blueprint('http', __name__)


@app.route('/process_image', methods=['POST'])
def process_image():

    config = get_config('path-to_config')

    segmentation_model = SegmentationModel(config)
    depth_model = DepthModel(config)

    image = get_image(request.files['image'])

    segmentation_results = segmentation_model.predict(image)

    depth_results = depth_model.predict(image, segmentation_results)

    strategy = SelectionStrategy(image, segmentation_results, depth_results)
    image_grab_point, label = strategy.select()

    return jsonify({'point': image_grab_point.tolist(), 'label': label})


def get_config(path):
    pass

def get_image(data):
    image_np = np.fromstring(data.read())
    image = cv2.imdecode(image_np, np.uint8, cv2.IMREAD_UNCHANGED)
    return image


def save_image(image, path):
    pass


if __name__ == '__main__':
    app.register_blueprint(http_bp)
    app.register_blueprint(https_bp)
    app.run(host='0.0.0.0', port=5001)