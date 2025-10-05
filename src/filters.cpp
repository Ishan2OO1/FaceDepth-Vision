/*
 
  Bhumika Yadav , Ishan Chaudhary
  Fall 2025
  CS 5330 Computer Vision
 
  Image filtering implementations used by the apps. Includes custom
  grayscale and sepia, separable 5x5 blur (fast and naive), Sobel X/Y
  and per-channel magnitude, blur+quantize, and the four UI effects:
  blurOutsideFaces, emboss, median5x5, and colorPop.
 
*/
#include "filters.h"
#include "../faceDetect/faceDetect.h"
#include <iostream>
using namespace cv;

// Custom grayscale that inverts red channel for intensity
int greyscale(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return -1;
    }

    // Make sure dst has the same size/type as src
    dst.create(src.size(), src.type());

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // Get pixel (BGR format)
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            uchar blue  = pixel[0];
            uchar green = pixel[1];
            uchar red   = pixel[2];

            // Custom greyscale: invert red channel
            uchar intensity = 255 - red;

            // Assign same intensity to all channels
            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity, intensity);
        }
    }

    return 0;
}

// Sepia tone using linear transform per pixel
int sepia(Mat &src, Mat &dst) {
    if (src.empty()) return -1;

    dst = src.clone();  // same size/type as src

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // get original pixel (B, G, R order in OpenCV)
            Vec3b pixel = src.at<Vec3b>(y, x);
            uchar B = pixel[0];
            uchar G = pixel[1];
            uchar R = pixel[2];

            // compute new values using original values
            int newB = static_cast<int>(0.272 * R + 0.534 * G + 0.131 * B);
            int newG = static_cast<int>(0.349 * R + 0.686 * G + 0.168 * B);
            int newR = static_cast<int>(0.393 * R + 0.769 * G + 0.189 * B);

            // clamp to [0,255]
            newB = std::min(255, newB);
            newG = std::min(255, newG);
            newR = std::min(255, newR);

            // assign to destination pixel
            dst.at<Vec3b>(y, x) = Vec3b(newB, newG, newR);
        }
    }

    return 0;
}

// Naive 5x5 blur using "at" and integer approximation of Gaussian
// Naive 5x5 Gaussian-like blur (direct convolution)
int blur5x5_1(Mat &src, Mat &dst) {
    if (src.empty()) return -1;
    
    dst = src.clone(); // keep original edges intact

    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8,16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };
    int kernelSum = 1+2+4+2+1 + 2+4+8+4+2 + 4+8+16+8+4 + 2+4+8+4+2 + 1+2+4+2+1; // sum = 100

    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            Vec3i sum = {0,0,0}; // accumulate B,G,R
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    Vec3b pix = src.at<Vec3b>(y+ky, x+kx);
                    sum[0] += pix[0] * kernel[ky+2][kx+2];
                    sum[1] += pix[1] * kernel[ky+2][kx+2];
                    sum[2] += pix[2] * kernel[ky+2][kx+2];
                }
            }
            dst.at<Vec3b>(y, x)[0] = std::min(255, sum[0]/kernelSum);
            dst.at<Vec3b>(y, x)[1] = std::min(255, sum[1]/kernelSum);
            dst.at<Vec3b>(y, x)[2] = std::min(255, sum[2]/kernelSum);
        }
    }

    return 0;
}

// Fast separable 5x5 blur using two 1D passes
int blur5x5_2(Mat &src, Mat &dst) {
    if (src.empty()) return -1;
    
    dst = src.clone();

    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 10; // sum of 1+2+4+2+1

    // Use pointer access (ptr<>) to avoid repeated bounds-checking in at<>
    // To avoid double-integer-division rounding (which made the fast image darker),
    // store horizontal weighted sums in a CV_32SC3 temp matrix (no division), then
    // do the vertical weighted sum and divide once by 100 (the full kernel sum).
    int rows = src.rows;
    int cols = src.cols;

    Mat temp(rows, cols, CV_32SC3); // stores horizontal weighted integer sums

    // Horizontal pass: compute weighted sums (no normalization) into temp
    for (int y = 0; y < rows; y++) {
        const Vec3b* srcRow = src.ptr<Vec3b>(y);
        Vec3i* tempRow = temp.ptr<Vec3i>(y);

        // keep left/right two columns as in original by copying original values
        for (int x = 0; x < cols; x++) {
            tempRow[x] = Vec3i(0,0,0);
        }

        for (int x = 2; x < cols - 2; x++) {
            int b = 0, g = 0, r = 0;
            for (int k = -2; k <= 2; k++) {
                const Vec3b &p = srcRow[x + k];
                int w = kernel[k + 2];
                b += p[0] * w;
                g += p[1] * w;
                r += p[2] * w;
            }
            tempRow[x][0] = b; // store weighted sum (0..255*10)
            tempRow[x][1] = g;
            tempRow[x][2] = r;
        }
    }

    // Vertical pass: combine vertical weights with horizontal sums and divide by 100
    const int fullKernelSum = 100; // 10 * 10
    for (int y = 2; y < rows - 2; y++) {
        Vec3b* dstRow = dst.ptr<Vec3b>(y);
        for (int x = 0; x < cols; x++) {
            long b = 0, g = 0, r = 0;
            for (int k = -2; k <= 2; k++) {
                const Vec3i* tmpRow = temp.ptr<Vec3i>(y + k);
                const Vec3i &p = tmpRow[x];
                int w = kernel[k + 2];
                b += static_cast<long>(p[0]) * w;
                g += static_cast<long>(p[1]) * w;
                r += static_cast<long>(p[2]) * w;
            }
            int vb = static_cast<int>(b / fullKernelSum);
            int vg = static_cast<int>(g / fullKernelSum);
            int vr = static_cast<int>(r / fullKernelSum);
            dstRow[x][0] = static_cast<uchar>(std::min(255, std::max(0, vb)));
            dstRow[x][1] = static_cast<uchar>(std::min(255, std::max(0, vg)));
            dstRow[x][2] = static_cast<uchar>(std::min(255, std::max(0, vr)));
        }
    }

    return 0;
}

// Separable 3x3 Sobel X: kernel [[-1,0,1],[-2,0,2],[-1,0,1]]
// Separable Sobel X producing signed 16-bit output
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    int rows = src.rows;
    int cols = src.cols;

    // dst must be CV_16SC3 to hold negative values
    dst.create(rows, cols, CV_16SC3);

    // We implement separable as horizontal [ -1 0 1 ] then vertical [1 2 1]^T with signs
    // First horizontal pass: compute H = src * [ -1 0 1 ] for each channel into int temp
    Mat temp(rows, cols, CV_32SC3, Scalar(0,0,0));
    for (int y = 0; y < rows; y++) {
        const Vec3b* srow = src.ptr<Vec3b>(y);
        Vec3i* trow = temp.ptr<Vec3i>(y);
        for (int x = 1; x < cols - 1; x++) {
            const Vec3b &l = srow[x-1];
            const Vec3b &r = srow[x+1];
            trow[x][0] = static_cast<int>(r[0]) - static_cast<int>(l[0]);
            trow[x][1] = static_cast<int>(r[1]) - static_cast<int>(l[1]);
            trow[x][2] = static_cast<int>(r[2]) - static_cast<int>(l[2]);
        }
    }

    // Vertical pass: combine with [1 2 1]^T i.e. for each channel compute:
    // G = [1 2 1]^T * H
    for (int y = 1; y < rows - 1; y++) {
        Vec3s* drow = dst.ptr<Vec3s>(y);
        for (int x = 0; x < cols; x++) {
            const Vec3i* t0 = temp.ptr<Vec3i>(y-1);
            const Vec3i* t1 = temp.ptr<Vec3i>(y);
            const Vec3i* t2 = temp.ptr<Vec3i>(y+1);
            int b = t0[x][0] + 2*t1[x][0] + t2[x][0];
            int g = t0[x][1] + 2*t1[x][1] + t2[x][1];
            int r = t0[x][2] + 2*t1[x][2] + t2[x][2];
            // store in signed short
            drow[x][0] = static_cast<short>(std::max(-32768, std::min(32767, b)));
            drow[x][1] = static_cast<short>(std::max(-32768, std::min(32767, g)));
            drow[x][2] = static_cast<short>(std::max(-32768, std::min(32767, r)));
        }
    }

    return 0;
}

// Separable 3x3 Sobel Y: kernel [[1,2,1],[0,0,0],[-1,-2,-1]] (positive up)
// Separable Sobel Y producing signed 16-bit output
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    int rows = src.rows;
    int cols = src.cols;
    dst.create(rows, cols, CV_16SC3);

    // First vertical pass: compute V = src * [1 0 -1]^T (i.e., top - bottom) into int temp
    Mat temp(rows, cols, CV_32SC3, Scalar(0,0,0));
    for (int y = 1; y < rows - 1; y++) {
        const Vec3b* sup = src.ptr<Vec3b>(y-1);
        const Vec3b* sdown = src.ptr<Vec3b>(y+1);
        Vec3i* trow = temp.ptr<Vec3i>(y);
        for (int x = 0; x < cols; x++) {
            trow[x][0] = static_cast<int>(sup[x][0]) - static_cast<int>(sdown[x][0]);
            trow[x][1] = static_cast<int>(sup[x][1]) - static_cast<int>(sdown[x][1]);
            trow[x][2] = static_cast<int>(sup[x][2]) - static_cast<int>(sdown[x][2]);
        }
    }

    // Horizontal pass: combine with [1 2 1]
    for (int y = 0; y < rows; y++) {
        Vec3s* drow = dst.ptr<Vec3s>(y);
        const Vec3i* trow = temp.ptr<Vec3i>(y);
        for (int x = 1; x < cols - 1; x++) {
            int b = trow[x-1][0] + 2*trow[x][0] + trow[x+1][0];
            int g = trow[x-1][1] + 2*trow[x][1] + trow[x+1][1];
            int r = trow[x-1][2] + 2*trow[x][2] + trow[x+1][2];
            drow[x][0] = static_cast<short>(std::max(-32768, std::min(32767, b)));
            drow[x][1] = static_cast<short>(std::max(-32768, std::min(32767, g)));
            drow[x][2] = static_cast<short>(std::max(-32768, std::min(32767, r)));
        }
    }

    return 0;
}

// Compute color gradient magnitude: dst(x) = sqrt( sx(x)^2 + sy(x)^2 ) per channel
// Per-channel gradient magnitude from Sobel X/Y
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty()) return -1;
    if (sx.size() != sy.size()) return -1;
    if (sx.type() != CV_16SC3 || sy.type() != CV_16SC3) return -1;

    int rows = sx.rows;
    int cols = sx.cols;
    dst.create(rows, cols, CV_8UC3);

    for (int y = 0; y < rows; y++) {
        const Vec3s* sxrow = sx.ptr<Vec3s>(y);
        const Vec3s* syrow = sy.ptr<Vec3s>(y);
        Vec3b* drow = dst.ptr<Vec3b>(y);
        for (int x = 0; x < cols; x++) {
            int sb = sxrow[x][0];
            int sg = sxrow[x][1];
            int sr = sxrow[x][2];
            int tb = syrow[x][0];
            int tg = syrow[x][1];
            int tr = syrow[x][2];

            // compute magnitude per channel
            double mb = std::sqrt((double)sb*sb + (double)tb*tb);
            double mg = std::sqrt((double)sg*sg + (double)tg*tg);
            double mr = std::sqrt((double)sr*sr + (double)tr*tr);

            int ib = static_cast<int>(std::min(255.0, mb));
            int ig = static_cast<int>(std::min(255.0, mg));
            int ir = static_cast<int>(std::min(255.0, mr));

            drow[x][0] = static_cast<uchar>(ib);
            drow[x][1] = static_cast<uchar>(ig);
            drow[x][2] = static_cast<uchar>(ir);
        }
    }

    return 0;
}

// Blur the image (using separable 5x5 blur) and quantize each channel into 'levels' buckets
// Blur then quantize colors into given number of levels
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty()) return -1;
    if (levels < 1) levels = 1;

    // First blur into a temporary Mat (use our fast separable blur)
    cv::Mat blurred;
    blur5x5_2(src, blurred);

    dst.create(blurred.size(), blurred.type());

    // bucket size: b = 255 / levels; but to ensure last bucket reaches 255, compute as double
    double b = 255.0 / levels;

    for (int y = 0; y < blurred.rows; y++) {
        const Vec3b* brow = blurred.ptr<Vec3b>(y);
        Vec3b* drow = dst.ptr<Vec3b>(y);
        for (int x = 0; x < blurred.cols; x++) {
            Vec3b p = brow[x];
            Vec3b q;
            for (int c = 0; c < 3; c++) {
                int xt = static_cast<int>(p[c] / b);
                int xf = static_cast<int>(xt * b);
                if (xf > 255) xf = 255;
                q[c] = static_cast<uchar>(xf);
            }
            drow[x] = q;
        }
    }

    return 0;
}

// Blur the image everywhere except inside detected face boxes
// Blur background and add blue tint; keep detected faces sharp
int blurOutsideFaces(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;
    // detect faces
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    detectFaces(grey, faces);

    // create a blurred version of the whole image
    cv::Mat blurred;
    blur5x5_2(src, blurred);

    // build mask: 1 inside faces, 0 outside
    cv::Mat mask(src.rows, src.cols, CV_8U, cv::Scalar(0));
    for (const auto &r : faces) {
        cv::Rect rr = r & cv::Rect(0,0,src.cols, src.rows);
        if (rr.area() > 0) mask(rr).setTo(255);
    }

    dst.create(src.size(), src.type());
    // copy face areas from src, rest from blurred with stronger blue tint
    for (int y = 0; y < src.rows; y++) {
        const cv::Vec3b* srow = src.ptr<cv::Vec3b>(y);
        const cv::Vec3b* brow = blurred.ptr<cv::Vec3b>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        cv::Vec3b* drow = dst.ptr<cv::Vec3b>(y);
        for (int x = 0; x < src.cols; x++) {
            if (mrow[x]) {
                drow[x] = srow[x];
            } else {
                cv::Vec3b p = brow[x];
                // boost blue channel and slightly reduce red/green to emphasize blue background
                int b = std::min(255, (int)p[0] + 60);
                int g = std::max(0, (int)(p[1] * 0.85));
                int r = std::max(0, (int)(p[2] * 0.85));
                drow[x] = cv::Vec3b((uchar)b, (uchar)g, (uchar)r);
            }
        }
    }

    return 0;
}

// Emboss effect using Sobel X/Y: compute signed gradient and project onto direction
// Emboss via directional projection of Sobel gradients
int emboss(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;
    cv::Mat sx, sy;
    sobelX3x3(src, sx); // CV_16SC3
    sobelY3x3(src, sy);

    // direction vector (normalize)
    const float vx = 0.7071f, vy = 0.7071f; // diagonal light

    dst.create(src.size(), src.type());
    for (int y = 0; y < src.rows; y++) {
        const cv::Vec3s* sxrow = sx.ptr<cv::Vec3s>(y);
        const cv::Vec3s* syrow = sy.ptr<cv::Vec3s>(y);
        cv::Vec3b* drow = dst.ptr<cv::Vec3b>(y);
        for (int x = 0; x < src.cols; x++) {
            int b = sxrow[x][0] * vx + syrow[x][0] * vy;
            int g = sxrow[x][1] * vx + syrow[x][1] * vy;
            int r = sxrow[x][2] * vx + syrow[x][2] * vy;
            // shift to mid-gray and clamp
            int ib = std::min(255, std::max(0, 128 + b));
            int ig = std::min(255, std::max(0, 128 + g));
            int ir = std::min(255, std::max(0, 128 + r));
            drow[x] = cv::Vec3b((uchar)ib, (uchar)ig, (uchar)ir);
        }
    }

    return 0;
}

// Median filter 5x5 (area-based)
// Median smoothing with 5x5 kernel
int median5x5(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;
    cv::medianBlur(src, dst, 5);
    return 0;
}

// Color pop: keep target hue, desaturate others
// Preserve pixels near target hue, grayscale elsewhere
int colorPop(cv::Mat &src, cv::Mat &dst, int targetHueDeg, int hueToleranceDeg) {
    if (src.empty()) return -1;
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    dst = src.clone();
    int lower = (targetHueDeg - hueToleranceDeg + 180) % 180;
    int upper = (targetHueDeg + hueToleranceDeg) % 180;
    cv::Mat mask(hsv.rows, hsv.cols, CV_8U, cv::Scalar(0));
    for (int y = 0; y < hsv.rows; y++) {
        const cv::Vec3b* hrow = hsv.ptr<cv::Vec3b>(y);
        uchar* mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < hsv.cols; x++) {
            int h = hrow[x][0];
            bool inRange = (lower <= upper) ? (h >= lower && h <= upper)
                                            : (h >= lower || h <= upper);
            mrow[x] = inRange ? 255 : 0;
        }
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
    dst.setTo(cv::Scalar(0,0,0));
    gray.copyTo(dst);
    src.copyTo(dst, mask);
    return 0;
}

// Keep faces colorful and desaturate rest
int colorFaces(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;
    cv::Mat grey; cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces; detectFaces(grey, faces);
    cv::Mat gray3; cv::cvtColor(grey, gray3, cv::COLOR_GRAY2BGR);
    dst.create(src.size(), src.type());
    gray3.copyTo(dst);
    for (const auto &r : faces) {
        cv::Rect rr = r & cv::Rect(0,0,src.cols, src.rows);
        if (rr.area() > 0) src(rr).copyTo(dst(rr));
    }
    return 0;
}

// Pixelate background while keeping faces sharp
int pixelateBackground(cv::Mat &src, cv::Mat &dst, int blockSize) {
    if (src.empty()) return -1;
    if (blockSize < 4) blockSize = 4;
    cv::Mat grey; cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces; detectFaces(grey, faces);
    cv::Mat tiny, pixelated;
    cv::resize(src, tiny, cv::Size(std::max(1, src.cols / blockSize), std::max(1, src.rows / blockSize)), 0, 0, cv::INTER_LINEAR);
    cv::resize(tiny, pixelated, src.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat mask(src.rows, src.cols, CV_8U, cv::Scalar(0));
    for (const auto &r : faces) {
        cv::Rect rr = r & cv::Rect(0,0,src.cols, src.rows);
        if (rr.area() > 0) mask(rr).setTo(255);
    }
    dst.create(src.size(), src.type());
    pixelated.copyTo(dst);
    src.copyTo(dst, mask);
    return 0;
}
