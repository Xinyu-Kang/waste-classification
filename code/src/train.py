from ultralytics_rgbd.ultralytics.models.yolo import YOLO
import torch
import sys 

if __name__ == "__main__":
    sys.path.append('/home/xinyukang/ZZDance2.0/code/src/ultralytics_rgbd')
    # model = YOLO(model="code/checkpoints/segmentation/best-updated.pt", verbose=True)
    model = YOLO(model="../checkpoints/segmentation/best-updated.pt", verbose=True)
    model.train(data="../../../../card_data/data.yaml", epochs=100, imgsz=640)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
