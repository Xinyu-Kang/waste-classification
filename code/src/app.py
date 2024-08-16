from flask import Flask, request, jsonify
import cv2
import numpy as np

from segmentation import SegmentationModel # 分割模型，YOLO
from depth import DepthModel 
from strategy import SelectionStrategy

app = Flask(__name__) 

segmentation_model = SegmentationModel("yolo")
depth_model = DepthModel("depth-anything")
strategy = SelectionStrategy()


@app.route('/process_image', methods=['POST'])
def process_image():

    image = get_image(request.data)

    segmentation_results = segmentation_model.predict(image)

    depth_results = depth_model.predict(image, segmentation_results)

    image_point, label = strategy.select(depth_results)

    
    return jsonify({"grab_point": image_point.tolist(), "label": label})


def get_image(data):
    pass
