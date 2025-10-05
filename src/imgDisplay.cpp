/*
 
  Bhumika Yadav , Ishan Chaudhary
  Fall 2025
  CS 5330 Computer Vision
 
  Displays a single test image in a window and waits for a keypress.
  Useful for verifying OpenCV installation and basic I/O before running
  the live video and filtering demos.
 
*/
#include <opencv2/opencv.hpp>
#include <iostream>

// Displays a single image and waits for user keypress
int main() {
    // Local image path
    std::string imagePath = "../test.jpg";

    // Load the image
    cv::Mat img = cv::imread(imagePath);

    if (img.empty()) {
        std::cout << "Could not open or find the image: " << imagePath << std::endl;
        return -1;
    }

    // Create a window
    cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);

    // Show the image
    cv::imshow("Display Window", img);

    std::cout << "Press 'q' to quit." << std::endl;

    // Event loop for key press
    while (true) {
        char key = (char)cv::waitKey(0); // waits indefinitely for a key press
        if (key == 'q' || key == 'Q') break;
        else std::cout << "You pressed: " << key << std::endl;
    }

    // Destroy the window
    cv::destroyAllWindows();

    return 0;
}
