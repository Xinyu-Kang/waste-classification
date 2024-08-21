import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

class SegmentationModel:

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model = None
        self.load()
    
    def load(self):
        """
        根据模型名称加载模型。
        """
        # 假设模型文件存放在 './checkpoints/segmentation/' 目录下
        model_path = f'../checkpoints/segmentation/{self.model_name}.pt'
        if os.path.exists(model_path):
            self.model = YOLO(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            raise ValueError(f"Model file not found: {model_path}")
        
    def get_label_names(self):
        model_names = self.model.names
        return model_names

    def predict(self, image):
        """
        对输入图像进行分割预测。
        :param image: 输入图像（通常是 NumPy 数组）。
        :return: 直接返回 YOLO 模型的预测结果。
        """
        if self.model is None:
            raise RuntimeError("Model not loaded, please call load() before predict()")
        
        height, width = image.shape[:2]

        # 如果是 YOLO 模型，输入必须是一个格式化的图像
        results = self.model(image,conf=0.1, imgsz=(height, width), agnostic_nms=True)  # YOLO 模型可以直接接受 NumPy 数组
        
        return results
