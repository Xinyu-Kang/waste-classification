import yaml
import torch

if __name__ == "__main__":

    # Load the pre-trained checkpoint
    checkpoint_path = 'yolov8n.pt'
    checkpoint = torch.load(checkpoint_path)

    # Get the state dict
    state_dict = checkpoint['model'].state_dict()

    # Get the original weights for the first layer
    original_weights = state_dict['model.0.conv.weight']

    # Create a zero tensor for the additional channel
    zero_channel = torch.zeros((original_weights.shape[0], 1, original_weights.shape[2], original_weights.shape[3]))

    # Concatenate the original weights with the zero channel
    new_weights = torch.cat((original_weights, zero_channel), dim=1)

    # Update the state dictionary with the new weights
    state_dict['model.0.conv.weight'] = new_weights

    # Update the first layer for 4-channel inputs
    # Use the original number of output channels
    num_output_channels = original_weights.shape[0]
    checkpoint['model'].model[0].conv = torch.nn.Conv2d(4, num_output_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    # Update the checkpoint with the modified state dictionary
    checkpoint['model'].load_state_dict(state_dict, strict=False)

    # Save the modified checkpoint
    updated_checkpoint_path = 'yolov8n_updated.pt'
    torch.save(checkpoint, updated_checkpoint_path)