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

segmentation_model = SegmentationModel("yolo")
depth_model = DepthModel("depth-anything")
strategy = SelectionStrategy()


@app.route('/process_image', methods=['POST'])
def process_image():

    image = get_image(request.data)

    segmentation_results = segmentation_model.predict(image)

    depth_results = depth_model.predict(image, segmentation_results)

    image_point, label = strategy.select(segmentation_results, depth_results)

    
    return jsonify({"grab_point": image_point.tolist(), "label": label})


def get_image(data):
    pass

if __name__ == '__main__':
    app.register_blueprint(http_bp)
    app.register_blueprint(https_bp)
    app.run(host="0.0.0.0", port=5001)