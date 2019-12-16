cmake_minimum_required(VERSION 3.6)
cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo")

project(SLAM)

set(CMAKE_MACOSX_RPATH OFF)

# add extra include directories
#include_directories(/usr/local/include)

# add extra lib directories
#link_directories(/usr/local/lib)

# g2o
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/g2o)

# add shaders files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/camera.vs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/camera.fs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/frustum.fs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/frustum.vs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(G2O_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/g2o)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/g2o/cmake_modules)

find_package(OpenGL REQUIRED)
find_library(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(G2O REQUIRED)
add_definitions(${EIGEN3_DEFINITIONS})
include_directories( ${EIGEN3_INCLUDE_DIRS}) 
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLFW_INCLUDE_DIR})

message(${OpenCV_INCLUDE_DIRS})

set(GLEW_LIBRARY, /usr/local/lib/libglfw.dylib)
set(GLFW_LIBRARY, /usr/local/lib/libGLEW.dylib)
add_executable(SLAM main.cc Init.cc keyframe.cc map.cc optimizer.cc)

target_link_libraries(SLAM
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${GLEW_LIBRARY}
    ${GLFW_LIBRARY}
    ${CMAKE_CURRENT_SOURCE_DIR}/g2o/lib/libg2o_core.dylib
    ${CMAKE_CURRENT_SOURCE_DIR}/g2o/lib/libg2o_solver_csparse.dylib
    ${CMAKE_CURRENT_SOURCE_DIR}/g2o/lib/libg2o_stuff.dylib
    ${CMAKE_CURRENT_SOURCE_DIR}/g2o/lib/libg2o_csparse_extension.dylib
    ${CMAKE_CURRENT_SOURCE_DIR}/g2o/lib/libg2o_types_sba.dylib
)

