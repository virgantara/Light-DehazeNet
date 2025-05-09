import argparse
import cv2
import torch
import numpy as np
import time
from CVR import cvr

def lightdehaze_video(input_path, output_path, model_path, width=224, height=224):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Cannot open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width * 2, height)  # side-by-side layout
    )

    print(f" Input: {input_path}, Resolution: {width}x{height}, Total frames: {total_frames}")

    # Load model
    print(" Loading TorchScript model...")
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    start_time = time.time()
    frame_count = 0

    skip_frame = 5  # <-- dehaze every N frames
    last_dehazed = None  # cached result

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (width, height))
        if frame_count % skip_frame == 0:
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                dehaze_tensor = model(input_tensor)

            dehaze_np = dehaze_tensor.squeeze().cpu().numpy()
            dehaze_np = np.transpose(dehaze_np, (1, 2, 0))
            dehaze_np = (dehaze_np * 255).clip(0, 255).astype(np.uint8)
            dehaze_bgr = cv2.cvtColor(dehaze_np, cv2.COLOR_RGB2BGR)

            last_dehazed = dehaze_bgr
            # last_dehazed = cvr(dehaze_bgr)  # apply CVR only when updated
        # rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

        # with torch.no_grad():
        #     dehaze_tensor = model(input_tensor)

        # # Postprocess
        # dehaze_np = dehaze_tensor.squeeze().cpu().numpy()
        # dehaze_np = np.transpose(dehaze_np, (1, 2, 0))
        # dehaze_np = (dehaze_np * 255).clip(0, 255).astype(np.uint8)
        # dehaze_bgr = cv2.cvtColor(dehaze_np, cv2.COLOR_RGB2BGR)

        # # CVR enhancement
        # enhanced = cvr(dehaze_bgr)

        # Combine original and dehazed
        combined = cv2.hconcat([frame_resized, last_dehazed])

        # FPS + ETA overlay
        elapsed = time.time() - start_time
        avg_fps = (frame_count + 1) / elapsed
        eta = (total_frames - frame_count - 1) / avg_fps if avg_fps > 0 else 0
        cv2.putText(combined, f"FPS: {avg_fps:.2f} | ETA: {int(eta)}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Original | Dehazed+CVR", combined)
        out.write(combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_count} frames in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video dehazing using LightDehazeNet + CVR")
    parser.add_argument("-i", "--input", required=True, help="Path to input video (or 0 for webcam)")
    parser.add_argument("-o", "--output", required=True, help="Output video filename")
    parser.add_argument("-m", "--model", default="lightdehaze_jit.pt", help="TorchScript model path")
    parser.add_argument("--width", type=int, default=640, help="Resize width per frame")
    parser.add_argument("--height", type=int, default=480, help="Resize height per frame")
    args = parser.parse_args()

    input_path = int(args.input) if args.input == "0" else args.input
    lightdehaze_video(input_path, args.output, args.model, args.width, args.height)
