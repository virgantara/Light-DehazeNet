import cv2
import argparse
import time
from cap_dehaze import cap_dehaze

def dehaze_video(input_path, output_path, width=640, height=360, frameskip=5):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(" Error: Cannot open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(
        output_path,
        # cv2.VideoWriter_fourcc(*"XVID"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height)  # double width for side-by-side output
    )

    print(f"Input: {input_path}, Total frames: {total_frames}, Target size: {width}x{height} (each view)")

    frame_count = 0
    start_time = time.time()

    skip_frame = frameskip  # <-- dehaze every N frames
    last_dehazed = None  # cached result

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (width, height))
        frame_start = time.time()

        if frame_count % skip_frame == 0:
            dehazed = cap_dehaze(frame_resized)
        
        frame_end = time.time()

        # Combine frames side-by-side
        combined = cv2.hconcat([frame_resized, dehazed])

        latency_ms = (frame_end - frame_start) * 1000

        # Add FPS & ETA overlay
        elapsed = frame_end - start_time
        avg_fps = (frame_count + 1) / elapsed
        remaining_frames = total_frames - (frame_count + 1)
        eta = remaining_frames / avg_fps if avg_fps > 0 else 0

        cv2.putText(
            combined,
            f"FPS: {avg_fps:.2f} | ETA: {int(eta)}s | Latency {latency_ms:.1f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Original | CAP Dehazed", combined)
        out.write(combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({frame_count / total_time:.2f} FPS).")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dehaze video using Color Attenuation Prior (CAP) with side-by-side comparison")
    parser.add_argument("-i", "--input", required=True, help="Path to input video or 0 for webcam")
    parser.add_argument("-o", "--output", required=True, help="Path to output video (e.g., output.avi)")
    parser.add_argument("--width", type=int, default=640, help="Resize width per view (original and dehazed)")
    parser.add_argument("--height", type=int, default=360, help="Resize height")
    parser.add_argument("--frameskip", type=int, default=5, help="Resize height")
    args = parser.parse_args()

    input_path = int(args.input) if args.input == "0" else args.input
    dehaze_video(input_path, args.output, args.width, args.height, args.frameskip)
