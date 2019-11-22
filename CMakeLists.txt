cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework Cocoa -framework OpenGL -framework IOKit")

project(SLAM)

# add extra include directories
include_directories(/usr/local/include)

# add extra lib directories
link_directories(/usr/local/lib)

# g2o
include_directories(/usr/local/opt/g2o/include)
link_directories(/usr/local/opt/g2o/lib)

# add shaders files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/camera.vs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/camera.fs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/frustum.fs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/frustum.vs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(SLAM main.cc Init.cc keyframe.cc map.cc optimizer.cc)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(G2O REQUIRED)
add_definitions     ( ${EIGEN3_DEFINITIONS} )
include_directories ( ${EIGEN3_INCLUDE_DIRS} )
include_directories("${OpenCV_INCLUDE_DIRS}" "${PROJECT_BINARY_DIR}")

target_link_libraries(SLAM ${OpenCV_LIBS} glfw glew ${EIGEN3_LIBS} g2o_core g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba)
