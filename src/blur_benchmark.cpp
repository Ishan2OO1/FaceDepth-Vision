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

    cv::Mat out1, out2;
    const int runs = 20;

    // Warmup
    blur5x5_1(src, out1);
    blur5x5_2(src, out2);

    // Time naive
    double t0 = (double)cv::getTickCount();
    for (int i = 0; i < runs; ++i) {
        blur5x5_1(src, out1);
    }
    double t1 = (double)cv::getTickCount();
    double sec1 = (t1 - t0) / cv::getTickFrequency();

    // Time fast
    t0 = (double)cv::getTickCount();
    for (int i = 0; i < runs; ++i) {
        blur5x5_2(src, out2);
    }
    t1 = (double)cv::getTickCount();
    double sec2 = (t1 - t0) / cv::getTickFrequency();

    std::cout << "Naive blur average time: " << (sec1 / runs) << " sec/frame" << std::endl;
    std::cout << "Fast blur average time:  " << (sec2 / runs) << " sec/frame" << std::endl;

    // Save outputs
    cv::imwrite("../capture_benchmark_naive.jpg", out1);
    cv::imwrite("../capture_benchmark_fast.jpg", out2);

    // Compute mean intensities for quick brightness check
    cv::Scalar meanNaive = cv::mean(out1);
    cv::Scalar meanFast = cv::mean(out2);
    std::cout << "Mean naive BGR: " << meanNaive << std::endl;
    std::cout << "Mean fast  BGR: " << meanFast << std::endl;

    // Compute max absolute difference between images
    cv::Mat diff;
    cv::absdiff(out1, out2, diff);
    cv::Mat diffGray;
    cv::cvtColor(diff, diffGray, cv::COLOR_BGR2GRAY);
    double minVal, maxVal;
    cv::minMaxLoc(diffGray, &minVal, &maxVal);
    std::cout << "Max per-pixel absolute difference (grayscale): " << maxVal << std::endl;

    return 0;
}
