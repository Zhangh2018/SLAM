cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework Cocoa -framework OpenGL -framework IOKit")

project(SLAM)

# add extra include directories
include_directories(/usr/local/include)

# add extra lib directories
link_directories(/usr/local/lib)

# add shaders files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/camera.vs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/camera.fs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/frustum.fs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/frustum.vs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(SLAM main.cc Init.cc keyframe.cc map.cc)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}" "${PROJECT_BINARY_DIR}")

target_link_libraries(SLAM ${OpenCV_LIBS} glfw glew)
