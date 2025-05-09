import torchvision
from PIL import Image
import argparse
from CVR import cvr 
import torch
import numpy as np
import cv2
import time


def single_dehaze_test(input_path, jit_model_path="lightdehaze_jit.pt"):
    start_total = time.time()

    # Load TorchScript model
    start = time.time()
    model = torch.jit.load(jit_model_path, map_location="cpu")
    model.eval()
    print(f"[Time] Load model: {time.time() - start:.3f} sec")

    # Load image and convert to tensor
    start = time.time()
    hazy_input_image = Image.open(input_path).convert("RGB")

    # Resize ben cilik demi fast inference
    hazy_input_image = hazy_input_image.resize((224, 224))  # or cv2.resize(...)

    hazy_np = np.asarray(hazy_input_image).astype(np.float32) / 255.0
    hazy_tensor = torch.from_numpy(hazy_np).permute(2, 0, 1).unsqueeze(0)
    print(f"[Time] Load and preprocess image: {time.time() - start:.3f} sec")

    # Inference
    start = time.time()
    with torch.no_grad():
        dehaze_tensor = model(hazy_tensor)
    print(f"[Time] Inference: {time.time() - start:.3f} sec")

    # Postprocess
    start = time.time()
    dehaze_np = dehaze_tensor.squeeze().cpu().numpy()
    dehaze_np = np.transpose(dehaze_np, (1, 2, 0))  # CHW → HWC
    dehaze_np = (dehaze_np * 255).clip(0, 255).astype(np.uint8)
    dehaze_bgr = cv2.cvtColor(dehaze_np, cv2.COLOR_RGB2BGR)
    print(f"[Time] Postprocess: {time.time() - start:.3f} sec")

    # Apply CVR
    start = time.time()
    enhanced_bgr = cvr(dehaze_bgr)
    print(f"[Time] CVR Enhancement: {time.time() - start:.3f} sec")

    # Save image
    start = time.time()
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    enhanced_tensor = torch.from_numpy(enhanced_rgb).permute(2, 0, 1).float() / 255.0
    torchvision.utils.save_image(enhanced_tensor, "dehaze_with_cvr.jpg")
    print(f"[Time] Save image: {time.time() - start:.3f} sec")

    print(f"[Total Execution Time]: {time.time() - start_total:.3f} sec")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-m", "--model", default="lightdehaze_jit.pt", help="path to TorchScript model")
    args = vars(ap.parse_args())

    single_dehaze_test(args["image"], args["model"])
