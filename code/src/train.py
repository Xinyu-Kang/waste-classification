from ultralytics_rgbd.ultralytics.models.yolo import YOLO
import torch
import sys 

if __name__ == "__main__":
    sys.path.append('/home/xinyukang/ZZDance2.0/code/src/ultralytics_rgbd')
    # model = YOLO(model="code/checkpoints/segmentation/best-updated.pt", verbose=True)
    model = YOLO(model="../checkpoints/segmentation/best-updated.pt", verbose=True)
    model.train(data="../../../../only180_data/dataset.yaml",
                epochs=120, batch=32, dropout=0.1, patience=50, cos_lr=True,
                imgsz=640, verbose=True, plots=True)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
