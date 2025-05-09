#include "CAP.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./dehaze <video_path>" << std::endl;
        return -1;
    }

    std::string video_path = argv[1];
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return -1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = (int)cap.get(cv::CAP_PROP_FPS);

    std::cout << "Input video: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;

    cv::Mat frame, dehazed, side_by_side;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto start = std::chrono::high_resolution_clock::now();
        dehazing_CAP(frame, dehazed);
        auto end = std::chrono::high_resolution_clock::now();

        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double current_fps = 1000.0 / latency_ms;

        // Fix shape and type before concatenation
        if (dehazed.size() != frame.size()) {
            cv::resize(dehazed, dehazed, frame.size());
        }
        if (dehazed.type() != frame.type()) {
            dehazed.convertTo(dehazed, frame.type(), 255.0);
        }

        cv::hconcat(frame, dehazed, side_by_side);

        std::string text = cv::format("FPS: %.2f | Latency: %.1f ms", current_fps, latency_ms);
        cv::putText(side_by_side, text, cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Video Dehazing (CAP)", side_by_side);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
