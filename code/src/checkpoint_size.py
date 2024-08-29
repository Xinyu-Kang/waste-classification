import yaml
import torch

if __name__ == "__main__":

    checkpoint_path = f'../../yolov8n.pt'
    # checkpoint_path = f'../checkpoints/segmentation/best-updated.pt'
    checkpoint = torch.load(checkpoint_path)

    state_dict = checkpoint['model'].state_dict()
    weights = state_dict['model.0.conv.weight']
    print(weights.shape)

