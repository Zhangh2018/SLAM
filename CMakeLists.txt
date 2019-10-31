cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)

project(SLAM)
add_executable(SLAM main.cc Init.cc)
find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}" "${PROJECT_BINARY_DIR}")
target_link_libraries(SLAM ${OpenCV_LIBS})
