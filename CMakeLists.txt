cmake_minimum_required(VERSION 3.10)
project(ComputerVisionProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)
include_directories(faceDetect)
include_directories(da2-code)

# Add ONNX Runtime (if available)
# Note: You may need to adjust this path based on your ONNX Runtime installation
# find_package(onnxruntime REQUIRED)

# Source files
set(SOURCES
    src/filters.cpp
    src/imgDisplay.cpp
    src/vidDisplay.cpp
    faceDetect/faceDetect.cpp
)

# Header files
set(HEADERS
    src/filters.h
    faceDetect/faceDetect.h
    da2-code/DA2Network.hpp
)

# Executables
add_executable(blur_benchmark src/blur_benchmark.cpp ${SOURCES})
add_executable(sobel_test src/sobel_test.cpp ${SOURCES})
add_executable(imgDisplay src/imgDisplay.cpp ${SOURCES})
add_executable(vidDisplay src/vidDisplay.cpp ${SOURCES})

# Depth Anything v2 demo (requires ONNX Runtime)
# Uncomment when ONNX Runtime is properly configured
# add_executable(da2_demo src/da2_demo.cpp ${SOURCES})

# Link OpenCV
target_link_libraries(blur_benchmark ${OpenCV_LIBS})
target_link_libraries(sobel_test ${OpenCV_LIBS})
target_link_libraries(imgDisplay ${OpenCV_LIBS})
target_link_libraries(vidDisplay ${OpenCV_LIBS})

# Link ONNX Runtime (when available)
# target_link_libraries(da2_demo ${OpenCV_LIBS} onnxruntime)

# Copy necessary files to build directory
configure_file(${CMAKE_SOURCE_DIR}/faceDetect/haarcascade_frontalface_alt2.xml 
               ${CMAKE_BINARY_DIR}/faceDetect/haarcascade_frontalface_alt2.xml COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/data/cathedral.jpeg 
               ${CMAKE_BINARY_DIR}/data/cathedral.jpeg COPYONLY)

# Copy ONNX model (when available)
# configure_file(${CMAKE_SOURCE_DIR}/da2-code/model_fp16.onnx 
#                ${CMAKE_BINARY_DIR}/da2-code/model_fp16.onnx COPYONLY)

# Set output directory
set_target_properties(blur_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(sobel_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(imgDisplay PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(vidDisplay PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Compiler-specific options
if(MSVC)
    target_compile_options(blur_benchmark PRIVATE /W4)
    target_compile_options(sobel_test PRIVATE /W4)
    target_compile_options(imgDisplay PRIVATE /W4)
    target_compile_options(vidDisplay PRIVATE /W4)
else()
    target_compile_options(blur_benchmark PRIVATE -Wall -Wextra -pedantic)
    target_compile_options(sobel_test PRIVATE -Wall -Wextra -pedantic)
    target_compile_options(imgDisplay PRIVATE -Wall -Wextra -pedantic)
    target_compile_options(vidDisplay PRIVATE -Wall -Wextra -pedantic)
endif()
