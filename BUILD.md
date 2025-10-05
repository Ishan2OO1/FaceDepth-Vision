# Build Instructions

This document provides detailed instructions for building and running the Computer Vision project.

## Prerequisites

### Required Dependencies
- **C++17 compatible compiler** (Visual Studio 2019+, GCC 7+, or Clang 5+)
- **CMake 3.10+**
- **OpenCV 4.x** (Download from [OpenCV Releases](https://github.com/opencv/opencv/releases))
- **ONNX Runtime** (Download from [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime/releases))

> **Note**: This repository contains only the essential source code. Large dependencies like OpenCV and ONNX Runtime must be installed separately following the links above.

### Platform-Specific Setup

#### Windows (Visual Studio)
```bash
# Install Visual Studio 2019 or later with C++ development tools
# Install CMake from https://cmake.org/download/
# Install OpenCV 4.x and set OPENCV_DIR environment variable
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install build-essential cmake
sudo apt install libopencv-dev libopencv-contrib-dev
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install dependencies via Homebrew
brew install cmake opencv
```

## Building the Project

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Project-1
mkdir build
cd build
```

### 2. Configure with CMake
```bash
# Basic configuration
cmake ..

# With specific OpenCV path (if needed)
cmake -DOpenCV_DIR=/path/to/opencv/build ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### 3. Build
```bash
# Build all targets
cmake --build .

# Build specific target
cmake --build . --target blur_benchmark
cmake --build . --target sobel_test
cmake --build . --target imgDisplay
cmake --build . --target vidDisplay
```

### 4. Run Executables
```bash
# From build directory
./bin/blur_benchmark
./bin/sobel_test
./bin/imgDisplay
./bin/vidDisplay

# Depth Anything v2 demo (requires ONNX Runtime)
# ./bin/da2_demo
```

## ONNX Runtime Setup (Optional)

The depth estimation features require ONNX Runtime. Follow these steps:

### Windows
1. Download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases
   - **Recommended**: Download the NuGet package (microsoft.ml.onnxruntime.1.22.1.zip)
   - Extract to your project directory or system-wide location
2. Extract to a directory (e.g., `C:\onnxruntime` or project directory)
3. Set environment variables:
   ```cmd
   set ONNXRUNTIME_ROOT=C:\onnxruntime
   set PATH=%PATH%;%ONNXRUNTIME_ROOT%\lib
   ```

### Linux
```bash
# Download and extract ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ONNXRUNTIME_ROOT/lib
```

### macOS
```bash
# Install via Homebrew
brew install onnxruntime
```

## Troubleshooting

### Common Issues

#### OpenCV Not Found
```bash
# Set OpenCV path explicitly
cmake -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 ..
# or
cmake -DOpenCV_DIR=/path/to/opencv/build ..
```

#### ONNX Runtime Issues
- Ensure ONNX Runtime is properly installed
- Check that the model file `model_fp16.onnx` is in the correct location
- Verify environment variables are set correctly

#### Compilation Errors
- Ensure C++17 support is enabled
- Check that all required headers are available
- Verify OpenCV version compatibility

### Platform-Specific Notes

#### Windows
- Use Visual Studio Developer Command Prompt
- Ensure Windows SDK is installed
- May need to set `CMAKE_GENERATOR_PLATFORM` for 64-bit builds

#### Linux
- Install development packages: `sudo apt install build-essential`
- May need to install additional OpenCV dependencies
- Check library paths with `ldconfig -p | grep opencv`

#### macOS
- Ensure Xcode command line tools are installed
- May need to set `CMAKE_OSX_ARCHITECTURES` for Apple Silicon

## Testing the Build

### Basic Functionality Test
```bash
# Test blur benchmark
./bin/blur_benchmark

# Test Sobel edge detection
./bin/sobel_test

# Test image display
./bin/imgDisplay

# Test video display (requires camera)
./bin/vidDisplay
```

### Expected Output
- Blur benchmark should show performance comparison
- Sobel test should generate edge detection images
- Image display should show processed images
- Video display should show real-time camera feed with effects

## Development Setup

### IDE Configuration
- **Visual Studio Code**: Install C++ extension, configure CMake
- **Visual Studio**: Open folder, CMake integration
- **CLion**: Import CMake project
- **Qt Creator**: Open CMakeLists.txt

### Debugging
```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug

# Run with debugger
gdb ./bin/blur_benchmark
# or
lldb ./bin/blur_benchmark
```

## Performance Optimization

### Compiler Optimizations
```bash
# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### OpenCV Optimizations
- Enable OpenCV optimizations (IPP, TBB)
- Use optimized OpenCV builds when available
- Consider GPU acceleration for large images

## Continuous Integration

### GitHub Actions Example
```yaml
name: Build and Test
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Install dependencies
      run: |
        sudo apt update
        sudo apt install cmake libopencv-dev
    - name: Build
      run: |
        mkdir build && cd build
        cmake ..
        make
    - name: Test
      run: |
        cd build
        ./bin/blur_benchmark
```

---

For additional help, please refer to the main README.md or create an issue in the repository.
