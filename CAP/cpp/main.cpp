#include "CAP.h"
#include "opencv2/opencv.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Wrong usage. Usage: ./dehaze <image_path>" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    cv::Mat I = cv::imread(image_path, -1);
    if (I.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    cv::Mat J;
    dehazing_CAP(I, J);

    cv::imshow("Hazy Image", I);
    cv::imshow("Dehazed Image", J);
    cv::waitKey(0);
    return 0;
}
