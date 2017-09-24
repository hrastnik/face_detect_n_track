cmake_minimum_required(VERSION 3.6)
project(faceDetectionNTracking)

set(CMAKE_CXX_STANDARD 14)

file(COPY haarcascade_frontalface_default.xml DESTINATION ${PROJECT_BINARY_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp VideoFaceDetector.cpp VideoFaceDetector.h)
add_executable(demo ${SOURCE_FILES})

target_link_libraries(demo ${OpenCV_LIBS})