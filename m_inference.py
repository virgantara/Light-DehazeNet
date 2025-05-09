# @author: hayat
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import image_data_loader
import lightdehazeNet as ldnet
import numpy
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
from matplotlib import pyplot as plt


def image_haze_removel(input_image, model_path='trained_weights/trained_LDNet.pth'):

	
	hazy_image = (np.asarray(input_image)/255.0)

	hazy_image = torch.from_numpy(hazy_image).float().permute(2,0,1).unsqueeze(0)
	
	model = lightdehazeNet.LightDehaze_Net()

	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model.eval()

	with torch.no_grad():
		dehaze_image = model(hazy_image)
	
	return dehaze_image
	
		
# Optional TorchScript exporter for deployment
def export_to_torchscript(output_path="lightdehaze_jit.pt"):
    # model = lightdehazeNet.LightDehaze_Net()
    model = ldnet.LightDehazeNetLite()
    model.load_state_dict(torch.load('trained_weights/trained_LDNet.pth', map_location=torch.device('cpu')))
    model.eval()

    # Apply dynamic quantization (Linear, Conv2d)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    example_input = torch.rand(1, 3, 224, 224)  # Adjust if your input size is different
    scripted_model = torch.jit.trace(quantized_model, example_input)
    scripted_model.save(output_path)
    print(f"TorchScript model saved to {output_path}")


# Optional command-line export usage
if __name__ == "__main__":
    export_to_torchscript()