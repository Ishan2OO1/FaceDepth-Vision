/*
 
  Bhumika Yadav , Ishan Chaudhary
  Fall 2025
  CS 5330 Computer Vision
 
  Demonstrates Depth-Anything v2 inference. Streams webcam frames,
  runs the model to estimate relative depth, and optionally desaturates
  distant pixels while protecting detected faces. Includes temporal
  smoothing and median filtering controls.
 
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "../faceDetect/faceDetect.h"
#include "../da2-code/DA2Network.hpp"

// Depth-anything v2 demo: runs network, optional desaturation with depth
int main(int argc, char** argv) {
    // Model path relative to working dir (bin)
    const char *model_path = "../da2-code/model_fp16.onnx";

    // Create network (requires ONNX Runtime)
    DA2Network net(model_path);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Unable to open camera" << std::endl;
        return -1;
    }

    cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
    cv::Mat frame, depth;

    // Depth-desaturate controls
    bool desatMode = false; // toggle with 'd'
    int desat_thresh = 100; // depth value [0..255] where desaturation starts
    int desat_range = 120;  // over how many depth levels desaturation reaches full

    // Smoothing / filtering options
    bool smoothEnabled = true; // temporal smoothing enabled by default
    float smooth_alpha = 0.6f; // EMA factor (0..1) where higher = more smoothing
    cv::Mat depthAvg;          // CV_32F running average of depth
    bool medianFilter = true;  // apply median filter to depth

    // Face-protect option (keep detected faces in full color)
    bool faceProtect = true;
    float scale = 0.5f; // process at half resolution by default for speed

    std::cout << "Press q to quit. Press +/- to change processing scale (" << scale << ")" << std::endl;

    for(;;) {
        cap >> frame;
        if (frame.empty()) break;

        // prepare input for network and time the operation
        auto tstart = std::chrono::high_resolution_clock::now();
        net.set_input(frame, scale);

        // run network to produce depth at original frame size
        net.run_network(depth, frame.size());
        auto tend = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = tend - tstart;
        std::cout << "DA2 run time: " << elapsed.count() << " sec/frame" << std::endl;

        // depth is CV_8UC1 grayscale

        // Preprocess depth: optional temporal smoothing and median filtering
        cv::Mat depthProc;
        {
            // convert to float [0..1]
            cv::Mat depthF;
            depth.convertTo(depthF, CV_32F, 1.0 / 255.0);

            if (smoothEnabled) {
                if (depthAvg.empty()) {
                    depthAvg = depthF.clone();
                } else {
                    depthAvg = smooth_alpha * depthF + (1.0f - smooth_alpha) * depthAvg;
                }
                depthF = depthAvg;
            }

            // convert back to 8-bit for median filter and thresholding
            depthF.convertTo(depthProc, CV_8U, 255.0);

            if (medianFilter) {
                cv::medianBlur(depthProc, depthProc, 5);
            }
        }

        // If desaturation mode is active, blend the color frame with a grayscale version
        // based on the processed depth (farther pixels become desaturated)
        cv::Mat display;
        if (desatMode) {
            // normalize depthProc to [0..1]
            cv::Mat depthFN;
            depthProc.convertTo(depthFN, CV_32F, 1.0 / 255.0);

            // compute alpha mask where pixels with depth > thresh get desaturated
            float thr = desat_thresh / 255.0f;
            float rng = desat_range / 255.0f;
            cv::Mat alpha = (depthFN - thr) / (rng <= 0.0f ? 1.0f : rng);
            cv::threshold(alpha, alpha, 0.0, 0.0, cv::THRESH_TOZERO);
            cv::threshold(alpha, alpha, 1.0, 1.0, cv::THRESH_TRUNC);

            // face-protect: set alpha=0 inside detected face boxes
            if (faceProtect) {
                cv::Mat grey;
                cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
                std::vector<cv::Rect> faces;
                detectFaces(grey, faces);
                for (const auto &r : faces) {
                    cv::Rect rr = r & cv::Rect(0,0,alpha.cols, alpha.rows);
                    if (rr.area() > 0) {
                        alpha(rr).setTo(0.0f);
                    }
                }
            }

            // prepare color and gray images as float
            cv::Mat colorF, gray, gray3, grayF;
            frame.convertTo(colorF, CV_32F);
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);
            gray3.convertTo(grayF, CV_32F);

            // make alpha 3-channel
            cv::Mat alpha3;
            std::vector<cv::Mat> achan(3, alpha);
            cv::merge(achan, alpha3);

            // out = mix(color, gray, alpha)
            cv::Mat outF = colorF.mul(1.0f - alpha3) + grayF.mul(alpha3);
            outF.convertTo(display, CV_8U);
        } else {
            // when not desatMode, show the raw processed depth
            display = depthProc;
        }

        // overlay status text
        std::string status = std::string(desatMode ? "DESAT ON" : "DESAT OFF") +
                             (smoothEnabled ? " | SMOOTH" : " | NOSMOOTH") +
                             (medianFilter ? " | MED" : " | NOMED") +
                             (faceProtect ? " | FACEPROT" : "");
        cv::putText(display, status, cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);

        cv::imshow("Depth", display);

        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 'Q') break;
        else if (key == '+') { scale = std::min(1.0f, scale + 0.1f); std::cout << "scale=" << scale << std::endl; }
        else if (key == '-') { scale = std::max(0.2f, scale - 0.1f); std::cout << "scale=" << scale << std::endl; }
        else if (key == 's' || key == 'S') {
            std::string fn = "../capture_depth.jpg";
            if (cv::imwrite(fn, depth)) std::cout << "Saved depth to " << fn << std::endl;
            else std::cout << "Failed to save depth image" << std::endl;
        }
        else if (key == 'd' || key == 'D') {
            desatMode = !desatMode;
            std::cout << "Depth desaturate mode " << (desatMode ? "ON" : "OFF") << std::endl;
        }
        else if (key == 'c' || key == 'C') {
            // save the current color/processed view (if desatMode active, save desaturated color, else save original color)
            std::string fn = "../capture_desat.jpg";
            cv::Mat toSave;
            if (desatMode) {
                // recompute the desaturated color to save
                cv::Mat depthF; depth.convertTo(depthF, CV_32F, 1.0 / 255.0);
                float thr = desat_thresh / 255.0f;
                float rng = desat_range / 255.0f;
                cv::Mat alpha = (depthF - thr) / (rng <= 0.0f ? 1.0f : rng);
                cv::threshold(alpha, alpha, 0.0, 0.0, cv::THRESH_TOZERO);
                cv::threshold(alpha, alpha, 1.0, 1.0, cv::THRESH_TRUNC);
                cv::Mat colorF, gray, gray3, grayF;
                frame.convertTo(colorF, CV_32F);
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);
                gray3.convertTo(grayF, CV_32F);
                cv::Mat alpha3; std::vector<cv::Mat> achan(3, alpha); cv::merge(achan, alpha3);
                cv::Mat outF = colorF.mul(1.0f - alpha3) + grayF.mul(alpha3);
                outF.convertTo(toSave, CV_8U);
            } else {
                toSave = frame;
            }
            if (cv::imwrite(fn, toSave)) std::cout << "Saved desaturated capture to " << fn << std::endl;
            else std::cout << "Failed to save desaturated image" << std::endl;
        }
    }

    return 0;
}
