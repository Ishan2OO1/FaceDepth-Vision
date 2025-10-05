/*
 
  Bhumika Yadav , Ishan Chaudhary
  Fall 2025
  CS 5330 Computer Vision
 
  Declarations for image filtering functions used by the project, including
  grayscale, sepia, 5x5 blur (naive and separable), Sobel X/Y/magnitude,
  blur+quantize, and the selected interactive effects.
 
*/
// filters.h
#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

// Custom grayscale and classic transforms
int greyscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

// Selected effects in UI
int blurOutsideFaces(cv::Mat &src, cv::Mat &dst); // face-based
int emboss(cv::Mat &src, cv::Mat &dst);          // area-based
int median5x5(cv::Mat &src, cv::Mat &dst);       // area-based
int colorPop(cv::Mat &src, cv::Mat &dst, int targetHueDeg = 0, int hueToleranceDeg = 20); // pixel-wise
int colorFaces(cv::Mat &src, cv::Mat &dst);      // face regions in color, rest grayscale
int pixelateBackground(cv::Mat &src, cv::Mat &dst, int blockSize = 20); // pixelate non-face

#endif
