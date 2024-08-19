import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2

class DepthModel():
    def __init__(self, model_name) -> None:
        self.encoder = model_name
        self.model = None
        self.load()

    def load(self):
        self.load_depth_anything_v2()

    def load_depth_anything_v2(self):
        """
        根据encoder加载DepthAnythingV2模型
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        depth_anything_model = DepthAnythingV2(**model_configs[self.encoder])
        depth_anything_model.load_state_dict(torch.load(f'../checkpoints/depth_anything_v2_{self.encoder}.pth', map_location=device))
        depth_anything_model = depth_anything_model.to(device).eval()
        self.model = depth_anything_model

    def predict(self, image):
        """
        计算输入图像的深度图
        :param image: 输入图像
        :return: 深度图
        """
        depth_map = self.model.infer_image(image, input_size=518)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
        depth_map = depth_map.astype(np.uint8)
        return depth_map
