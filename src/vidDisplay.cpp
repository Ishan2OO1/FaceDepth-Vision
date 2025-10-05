/*
 
  Bhumika Yadav , Ishan Chaudhary
  Fall 2025
  CS 5330 Computer Vision
 
  Live video application with four interactive effects:
  Blur outside faces (with blue background tint), Emboss, Median 5x5,
  and Color Pop (red). Press 'o','e','r','P' to toggle effects, 's' to save,
  and 'c' for the original color feed.
 
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters.h"
#include "../faceDetect/faceDetect.h"

// Runs camera loop and toggles four effects via hotkeys
int main(int argc, char *argv[]) {
    cv::VideoCapture capdev(0);

    if (!capdev.isOpened()) {
        std::cerr << "Unable to open video device" << std::endl;
        return -1;
    }

    cv::Size refS((int) capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int) capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " x " << refS.height << std::endl;

    cv::namedWindow("Video", 1);
    cv::Mat frame, output;

    // Modes: full set of effects plus original color
    char mode = 'c';

    for (;;) {
        capdev >> frame;
        if (frame.empty()) {
            std::cerr << "Frame is empty" << std::endl;
            break;
        }

        double t0, t1, timeSec;

        switch(mode) {
            case 'g': // grayscale (OpenCV)
                cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
                cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
                break;

            case 'h': // custom grayscale
                greyscale(frame, output);
                break;

            case 'p': // sepia
                sepia(frame, output);
                break;

            case 'b': // fast 5x5 blur
                blur5x5_2(frame, output);
                break;

            case 'B': // naive 5x5 blur
                blur5x5_1(frame, output);
                break;

            case 'x': { // Sobel X (abs scaled)
                cv::Mat sx; sobelX3x3(frame, sx);
                cv::Mat abs8; cv::convertScaleAbs(sx, abs8, 255.0/1020.0);
                output = abs8;
                break;
            }

            case 'y': { // Sobel Y (abs scaled)
                cv::Mat sy; sobelY3x3(frame, sy);
                cv::Mat abs8; cv::convertScaleAbs(sy, abs8, 255.0/1020.0);
                output = abs8;
                break;
            }

            case 'm': { // gradient magnitude
                cv::Mat sx, sy, mag; sobelX3x3(frame, sx); sobelY3x3(frame, sy); magnitude(sx, sy, mag);
                output = mag; break;
            }

            case 'l': { // blur + quantize
                cv::Mat bq; blurQuantize(frame, bq, 10); output = bq; break;
            }

            case 'o': { // blur outside faces
                cv::Mat bo;
                blurOutsideFaces(frame, bo);
                output = bo;
                break;
            }

            case 'e': { // emboss
                cv::Mat em;
                emboss(frame, em);
                output = em;
                break;
            }

            case 'F': { // color faces only
                cv::Mat cf; colorFaces(frame, cf); output = cf; break;
            }

            case 'r': { // median 5x5 filter
                cv::Mat md;
                median5x5(frame, md);
                output = md;
                break;
            }

            case 'P': { // color pop (red hues kept)
                cv::Mat cp;
                colorPop(frame, cp, 0, 15);
                output = cp;
                break;
            }

            case 'k': { // pixelate background
                cv::Mat pb; pixelateBackground(frame, pb, 20); output = pb; break;
            }

            case 'z': { // face detection overlay
                cv::Mat grey; cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
                std::vector<cv::Rect> faces; detectFaces(grey, faces);
                drawBoxes(frame, faces);
                output = frame; break;
            }

            default: // color
                output = frame;
                break;
        }

        cv::imshow("Video", output);

        char key = (char)cv::waitKey(10);
        if (key == 'q' || key == 'Q') {
            std::cout << "Quitting..." << std::endl;
            break;
        }
        else if (key == 's' || key == 'S') {
            std::string filename;
            switch(mode) {
                case 'g': filename = "../capture_cvt_gray.jpg"; break;
                case 'h': filename = "../capture_custom_gray.jpg"; break;
                case 'p': filename = "../capture_sepia.jpg"; break;
                case 'b': filename = "../capture_fast_blur.jpg"; break;
                case 'B': filename = "../capture_naive_blur.jpg"; break;
                case 'x': filename = "../capture_sobel_x.jpg"; break;
                case 'y': filename = "../capture_sobel_y.jpg"; break;
                case 'm': filename = "../capture_magnitude.jpg"; break;
                case 'l': filename = "../capture_blur_quant.jpg"; break;
                case 'o': filename = "../capture_blur_outside_faces.jpg"; break;
                case 'e': filename = "../capture_emboss.jpg"; break;
                case 'F': filename = "../capture_color_faces.jpg"; break;
                case 'r': filename = "../capture_median5x5.jpg"; break;
                case 'P': filename = "../capture_color_pop.jpg"; break;
                case 'k': filename = "../capture_pixelate_background.jpg"; break;
                case 'z': filename = "../capture_face.jpg"; break;
                default:  filename = "../capture_color.jpg"; break;
            }
            if (cv::imwrite(filename, output))
                std::cout << "Saved frame to " << filename << std::endl;
            else
                std::cout << "Failed to save image!" << std::endl;
        }
        else if (key == 'c' || key == 'C') { mode = 'c'; std::cout << "Switched to color mode." << std::endl; }
        else if (key == 'g' || key == 'G') { mode = 'g'; std::cout << "Grayscale (OpenCV)." << std::endl; }
        else if (key == 'h' || key == 'H') { mode = 'h'; std::cout << "Custom grayscale." << std::endl; }
        else if (key == 'p') { mode = 'p'; std::cout << "Sepia." << std::endl; }
        else if (key == 'b') { mode = 'b'; std::cout << "Fast 5x5 blur." << std::endl; }
        else if (key == 'B') { mode = 'B'; std::cout << "Naive 5x5 blur." << std::endl; }
        else if (key == 'x' || key == 'X') { mode = 'x'; std::cout << "Sobel X." << std::endl; }
        else if (key == 'y' || key == 'Y') { mode = 'y'; std::cout << "Sobel Y." << std::endl; }
        else if (key == 'm' || key == 'M') { mode = 'm'; std::cout << "Gradient magnitude." << std::endl; }
        else if (key == 'l' || key == 'L') { mode = 'l'; std::cout << "Blur + quantize." << std::endl; }
        else if (key == 'o' || key == 'O') { mode = 'o'; std::cout << "Blur outside faces." << std::endl; }
        else if (key == 'e' || key == 'E') { mode = 'e'; std::cout << "Emboss." << std::endl; }
        else if (key == 'F') { mode = 'F'; std::cout << "Color faces only." << std::endl; }
        else if (key == 'r' || key == 'R') { mode = 'r'; std::cout << "Median 5x5." << std::endl; }
        else if (key == 'P') { mode = 'P'; std::cout << "Color pop (red)." << std::endl; }
        else if (key == 'k' || key == 'K') { mode = 'k'; std::cout << "Pixelate background." << std::endl; }
        else if (key == 'z' || key == 'Z') { mode = 'z'; std::cout << "Face detection overlay." << std::endl; }
    }

    return 0;
}
