# Computer Vision Project - Image Processing & Depth Estimation

**Authors:** Bhumika Yadav, Ishan Chaudhary  
**Course:** CS 5330 Computer Vision, Fall 2025  
**Institution:** Northeastern University  

## Overview

This project demonstrates advanced computer vision techniques including custom image filtering, face detection, depth estimation using deep learning, and real-time video processing. The implementation showcases both traditional computer vision algorithms and modern AI-based approaches for depth perception.

## Features

### 🖼️ Image Filtering & Effects
- **Custom Grayscale**: Inverted red channel grayscale conversion
- **Sepia Tone**: Classic sepia filter with linear transformations
- **Blur Algorithms**: Both naive 5x5 blur and optimized separable blur
- **Sobel Edge Detection**: X and Y gradient computation with magnitude calculation
- **Advanced Effects**: Emboss, median filtering, color quantization
- **Face-Aware Processing**: Blur outside faces, color pop with face protection

### 🧠 Deep Learning Integration
- **Depth Anything v2**: Real-time depth estimation using ONNX Runtime
- **Temporal Smoothing**: Exponential moving average for stable depth maps
- **Depth-Based Effects**: Selective desaturation based on depth information

### 👤 Face Detection
- **Haar Cascade**: Real-time face detection using OpenCV
- **Face Protection**: Preserve face regions during background processing
- **Multiple Effects**: Color faces while desaturating background

## Project Structure

```
Project-1/
├── src/                    # Source code
│   ├── filters.cpp         # Image filtering implementations
│   ├── filters.h           # Filter function declarations
│   ├── blur_benchmark.cpp  # Performance comparison
│   ├── sobel_test.cpp      # Edge detection testing
│   ├── da2_demo.cpp        # Depth Anything v2 demo
│   ├── imgDisplay.cpp      # Image display utilities
│   └── vidDisplay.cpp      # Video display utilities
├── da2-code/               # Depth Anything v2 integration
│   ├── DA2Network.hpp      # ONNX Runtime wrapper
│   ├── model_fp16.onnx     # Pre-trained depth model
│   └── *.cpp               # Example implementations
├── faceDetect/             # Face detection module
│   ├── faceDetect.cpp      # Haar cascade implementation
│   ├── faceDetect.h        # Face detection declarations
│   └── haarcascade_frontalface_alt2.xml
├── opencv/                 # OpenCV installation
├── bin/                    # Compiled executables
└── results/                # Output images and results
```

## Results Gallery

The project includes comprehensive results organized in the `results/` directory:

### Image Filtering Results (`results/filtering/`)
- **Grayscale Conversion**: Custom red-channel inversion (`capture_color.jpg`)
- **Sepia Tone**: Vintage photography effect (`capture_sepia.jpg`)
- **Blur Effects**: Naive vs optimized separable blur comparison (`capture_naive_blur.jpg`, `capture_blur_quant.jpg`)
- **Edge Detection**: Sobel X/Y gradients and magnitude (`capture_sobel_x.jpg`, `capture_sobel_y.jpg`, `capture_magnitude.jpg`)
- **Advanced Effects**: Emboss (`capture_emboss.jpg`), median filtering (`capture_median5x5.jpg`)

### Face Detection & Processing (`results/face_detection/`)
- **Face Detection**: Real-time face detection with bounding boxes (`capture_face.jpg`)
- **Face-Aware Blur**: Background blur while preserving faces (`capture_blur_outside_faces.jpg`)
- **Color Pop**: Selective colorization with face protection (`capture_color_pop.jpg`)
- **Depth-Based Effects**: Distance-based desaturation (`capture_desat.jpg`)

### Depth Estimation (`results/depth/`)
- **Real-time Depth**: Depth Anything v2 model inference (`capture_depth.jpg`)
- **Temporal Smoothing**: Stable depth maps with noise reduction
- **Depth Visualization**: Grayscale depth representation
- **Interactive Effects**: Depth-based selective processing

### Performance Benchmarks (`results/benchmarks/`)
- **Blur Performance**: Naive vs separable blur timing comparison (`capture_benchmark_naive.jpg`, `capture_benchmark_fast.jpg`)
- **Processing Speed**: Real-time performance metrics

## Technical Implementation

### Performance Optimizations
- **Separable Convolution**: 2-pass blur for O(n) complexity vs O(n²)
- **Pointer Arithmetic**: Efficient memory access patterns
- **Integer Approximation**: Fast Gaussian-like kernels
- **Temporal Smoothing**: EMA for stable real-time processing

### Deep Learning Pipeline
- **ONNX Runtime**: Cross-platform model inference
- **Dynamic Input**: Flexible image size handling
- **Normalization**: Proper input preprocessing for depth models
- **Output Scaling**: Depth map visualization and processing

### Face Detection Pipeline
- **Haar Cascades**: Robust frontal face detection
- **Multi-scale**: Detection at different image sizes
- **Histogram Equalization**: Improved detection accuracy
- **Bounding Box Processing**: Face region extraction and protection

## Prerequisites & Installation

### Required Software

#### 1. **OpenCV 4.x** - Computer Vision Library
- **Download**: [OpenCV Releases](https://github.com/opencv/opencv/releases)
- **Recommended Version**: OpenCV 4.8.0 or later
- **Installation Guides**:
  - **Windows**: [OpenCV Windows Installation](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html)
  - **Linux**: `sudo apt install libopencv-dev` (Ubuntu/Debian)
  - **macOS**: `brew install opencv` (using Homebrew)

#### 2. **ONNX Runtime** - Deep Learning Inference (Optional for Depth Features)
- **Download**: [Microsoft ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
- **Recommended Version**: 1.22.1 or compatible
- **Platform-specific packages**:
  - Windows: `onnxruntime-win-x64-1.22.1.zip`
  - Linux: `onnxruntime-linux-x64-1.22.1.tgz`
  - macOS: `onnxruntime-osx-x64-1.22.1.tgz`

#### 3. **Depth Anything v2 Model** - Pre-trained Depth Estimation Model
- **Download**: [Depth Anything v2 Model](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/tree/main)
- **File**: `model_fp16.onnx` (~47MB)
- **Place in**: `da2-code/` directory
- **Alternative**: Download from [official repository](https://github.com/DepthAnything/Depth-Anything-V2)

#### 4. **CMake** - Build System
- **Download**: [CMake Official Website](https://cmake.org/download/)
- **Minimum Version**: 3.10
- **Platform-specific**:
  - Windows: Download installer from cmake.org
  - Linux: `sudo apt install cmake` (Ubuntu/Debian)
  - macOS: `brew install cmake` (using Homebrew)

#### 5. **C++ Compiler**
- **Windows**: Visual Studio 2019+ or MinGW
- **Linux**: GCC 7+ or Clang 5+
- **macOS**: Xcode Command Line Tools (`xcode-select --install`)

### Quick Setup Scripts

#### Windows (PowerShell)
```powershell
# Install dependencies using package managers
winget install Kitware.CMake
winget install Microsoft.VisualStudio.2022.BuildTools

# Download OpenCV and ONNX Runtime manually from links above
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
# Download ONNX Runtime manually from releases page
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake opencv
# Download ONNX Runtime manually from releases page
```

## Building the Project

```bash
# Clone the repository
git clone <repository-url>
cd Project-1

# Build with CMake
mkdir build && cd build
cmake ..
make

# Run executables from bin/ directory
./bin/blur_benchmark.exe
./bin/da2_demo.exe
./bin/sobel_test.exe
```

## Usage Examples

### Image Filtering
```cpp
#include "filters.h"

cv::Mat src = cv::imread("image.jpg");
cv::Mat dst;

// Apply custom grayscale
greyscale(src, dst);

// Apply separable blur
blur5x5_2(src, dst);

// Face-aware blur
blurOutsideFaces(src, dst);
```

### Depth Estimation
```cpp
#include "DA2Network.hpp"

DA2Network net("model_fp16.onnx");
cv::Mat frame, depth;

net.set_input(frame, 0.5f);  // Half resolution for speed
net.run_network(depth, frame.size());
```

### Face Detection
```cpp
#include "faceDetect.h"

cv::Mat gray;
cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
std::vector<cv::Rect> faces;
detectFaces(gray, faces);
```

## Performance Metrics

- **Separable Blur**: ~3x faster than naive implementation
- **Depth Inference**: Real-time processing at 0.5x resolution
- **Face Detection**: 30+ FPS on standard hardware
- **Memory Efficiency**: Optimized pointer access patterns

## Key Algorithms

### Sobel Edge Detection
- Separable 3x3 kernels for X and Y gradients
- Signed 16-bit intermediate results
- Per-channel magnitude computation
- Proper edge handling and normalization

### Depth Estimation
- Depth Anything v2 model integration
- Dynamic input sizing with proper normalization
- Temporal smoothing with exponential moving average
- Median filtering for noise reduction

### Face-Aware Processing
- Haar cascade face detection
- Mask generation for face regions
- Selective processing based on detected faces
- Real-time performance optimization

## Future Enhancements

- [ ] GPU acceleration for depth inference
- [ ] Additional face detection algorithms (MTCNN, RetinaFace)
- [ ] More advanced depth-based effects
- [ ] Multi-threading for parallel processing
- [ ] Mobile platform support

## License

This project is developed for educational purposes as part of CS 5330 Computer Vision coursework.

## Acknowledgments

- **Depth Anything v2**: State-of-the-art depth estimation model
- **OpenCV Community**: Comprehensive computer vision library
- **ONNX Runtime**: Efficient model inference framework
- **Course Instructors**: Bruce A. Maxwell and teaching staff

---

*This project showcases the integration of traditional computer vision techniques with modern deep learning approaches, demonstrating both algorithmic understanding and practical implementation skills.*
