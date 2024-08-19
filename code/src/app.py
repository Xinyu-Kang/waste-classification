from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
import cv2
import numpy as np
import yaml
import queue
import ftplib
import asyncio
import aiofiles
from datetime import datetime

from segmentation import SegmentationModel 
from depth import DepthModel 
from strategy import SelectionStrategy

app = Flask(__name__) 
CORS(app)
https_bp = Blueprint('https', __name__)
http_bp = Blueprint('http', __name__)


# 队列用来存放从摄像头捕获的图片的二进制数据
image_queue = queue.Queue()

# ftp配置
# FTP配置
ftp_server = "183.222.62.175"
ftp_username = "hhl"
ftp_password = "GGhuman001"
remote_folder = "/RnD_CD/DATA_2024_08_19/"


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

async def upload_to_ftp(image_path, filename):
    try:
        async with aiofiles.open(image_path, 'rb') as file:
            data = await file.read()

        with ftplib.FTP(ftp_server) as ftp:
            ftp.login(ftp_username, ftp_password)
            ftp.cwd(remote_folder)
            ftp.storbinary(f'STOR {filename}', data)

        print(f"Successfully uploaded {filename} to FTP server.")
    except Exception as e:
        print(f"Failed to upload {filename} to FTP server: {e}")



@app.route('/process_image', methods=['POST'])
def process_image():
    """
    从request拿到现场图片,进行语义分割和深度预测,并使用策略选择抓取物体。
    :return:物体抓取点在图片上的坐标及物体的类别
    """
    image, filename = get_image(request.files['image'])

    # 异步上传图像到FTP服务器
    asyncio.create_task(upload_to_ftp(filename, filename))

    segmentation_results = segmentation_model.predict(image)
    label_names = segmentation_model.get_label_names()

    depth_map = depth_model.predict(image)

    strategy = SelectionStrategy(image, segmentation_results, depth_map, label_names)
    
    image_grab_point, label, points = strategy.select()
    print(f" point {image_grab_point}, label : {label}, object_img_pints: {points}")

    return jsonify({'point': image_grab_point.tolist(), 'label': label, 'object_img_pints': points})



def get_image(data):  
    # 将文件数据读取到一个字节流中
    image_np = np.frombuffer(data.read(), np.uint8)
    
    # 解码图像
    image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
    
    # 获取当前时间作为文件名
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    
    # 保存图像到本地
    # cv2.imwrite(filename, image)
    
    # 返回解码后的图像和文件名
    return image, filename



if __name__ == '__main__':
    app.register_blueprint(http_bp)
    app.register_blueprint(https_bp)
    app.run(host='0.0.0.0', port=5001)