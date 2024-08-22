from flask import Flask, request, jsonify, Blueprint, make_response, render_template, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import yaml
import queue
import ftplib
import asyncio
import aiofiles
from datetime import datetime
import os
import concurrent.futures

from draw import save_monitoring_image
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

# 监控图片目录
IMAGE_FOLDER = './monitoring'

# 确保目录存在
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)


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
    
    # 确保图像是三通道（RGB）
    if image.shape[-1] == 4:  # 如果图像是四通道（RGBA）
        image = image[:, :, :3]  # 去掉Alpha通道

        # 将上下30像素设置为黑色
    height, width, _ = image.shape

    # 上部30像素涂黑
    image[:30, :] = [0, 0, 0]  # 纯黑色 [B, G, R]，即[0, 0, 0]

    # 下部30像素涂黑
    image[-30:, :] = [0, 0, 0]  # 纯黑色


    # 异步上传图像到FTP服务器
    # asyncio.create_task(upload_to_ftp(filename, filename))

    segmentation_results = segmentation_model.predict(image)
    label_names = segmentation_model.get_label_names()



    # 如果分割模型没有返回结果，返回204 No Content
    if segmentation_results is None or not segmentation_results:
        return make_response('', 204)

    depth_map = depth_model.predict(image)

    # 归一化深度图，确保与第二段代码一致
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map = depth_map.astype(np.uint8)


    # 如果深度模型没有返回结果，返回204 No Content
    if depth_map is None or not depth_map.any():
        return make_response('', 204)

    strategy = SelectionStrategy(image, segmentation_results, depth_map, label_names)

    image_grab_point, label, points = strategy.select()

    # 如果抓取点、标签或points为空，返回204 No Content
    if image_grab_point is None or label is None or points is None:
        return make_response('', 204)


    # 使用线程池来异步保存监控图像
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(save_monitoring_image, image, segmentation_results, strategy.candidates, filename,  './monitoring')

   

    return jsonify({'point': image_grab_point, 'label': label, 'object_img_pints': points})


@app.route('/monitor')
def index():
    num_images = int(request.args.get('num_images', 10))
    images = sorted(os.listdir(IMAGE_FOLDER), key=lambda x: os.path.getmtime(os.path.join(IMAGE_FOLDER, x)), reverse=True)
    images = images[:num_images]
    return render_template('index.html', images=images)

@app.route('/image/<filename>')
def image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


def get_image(data):  
    # 将文件数据读取到一个字节流中
    image_np = np.frombuffer(data.read(), np.uint8)
    
    # 解码图像
    image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
    
    # 获取当前时间作为文件名
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    
    
    # 返回解码后的图像和文件名
    return image, filename



if __name__ == '__main__':
    # 在项目根目录下创建一个 'monitoring' 文件夹
    monitoring_dir = './monitoring'
    if not os.path.exists(monitoring_dir):
        os.makedirs(monitoring_dir)
    app.register_blueprint(http_bp)
    app.register_blueprint(https_bp)
    app.run(host='0.0.0.0', port=5001)