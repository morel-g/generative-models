import os
import torch
import torchvision.transforms as transforms
import torchaudio
import numpy as np
from PIL import Image

from src.data_manager.audio_data_manager import get_mel

"""
def tensor_to_image(tensor):
    # If the tensor has a channel dimension of size 3 (RGB), transpose it to (H, W, C)
    if len(tensor.shape) == 3 and tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # Apply the inverse transformation to get the values back to [0, 1]
    tensor = (tensor + 1) / 2
    
    # Clip values just in case they fall outside the [0, 1] range due to numerical reasons
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert tensor to a PIL Image
    img = transforms.ToPILImage()(tensor)
    
    return img
"""


def array_to_image(arr):
    # Ensure the input array is of type np.uint8
    if arr.dtype != np.uint8:
        # Check for single-channel (grayscale) images
        if arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)  # Remove the channel dimension
        # Check for three-channel (RGB) images
        elif arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))

        # Apply the inverse transformation to get the values back to [0, 1]
        arr = (arr + 1) / 2
        arr = np.clip(arr, 0, 1)

        arr = (arr * 255).astype(np.uint8)

    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(arr)

    return img


def imgs_to_audio(imgs, data_type, output_dir, name):
    mel = get_mel(data_type)

    for idx, img in enumerate(imgs):
        reconstruct_audio = mel.image_to_audio(array_to_image(img))
        audio_tensor = torch.tensor(reconstruct_audio).unsqueeze(0)
        path = os.path.join(output_dir, name + f"_audio_file_{idx}.wav")
        torchaudio.save(path, audio_tensor, sample_rate=22050)
