#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters.h"

int main() {
    std::string inPath = "../data/cathedral.jpeg";
    cv::Mat src = cv::imread(inPath);
    if (src.empty()) {
        std::cerr << "Failed to load " << inPath << std::endl;
        return -1;
    }

    cv::Mat sobelX16, sobelY16;
    if (sobelX3x3(src, sobelX16) != 0) {
        std::cerr << "sobelX3x3 failed" << std::endl;
        return -1;
    }
    if (sobelY3x3(src, sobelY16) != 0) {
        std::cerr << "sobelY3x3 failed" << std::endl;
        return -1;
    }

    // Visualize with convertScaleAbs
    cv::Mat sobelX8, sobelY8;
    cv::convertScaleAbs(sobelX16, sobelX8);
    cv::convertScaleAbs(sobelY16, sobelY8);

    cv::imwrite("../capture_sobelX_abs.jpg", sobelX8);
    cv::imwrite("../capture_sobelY_abs.jpg", sobelY8);

    // Save raw 16-bit signed mats to YAML for numeric inspection
    cv::FileStorage fsX("../capture_sobelX_16s.yml", cv::FileStorage::WRITE);
    fsX << "sobelX16" << sobelX16;
    fsX.release();

    cv::FileStorage fsY("../capture_sobelY_16s.yml", cv::FileStorage::WRITE);
    fsY << "sobelY16" << sobelY16;
    fsY.release();

    std::cout << "Saved visualized (abs) and raw (16s) Sobel outputs." << std::endl;
    return 0;
}
