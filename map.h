#ifndef MAP_H
#define MAP_H


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include "shader.h"
#include "keyframe.h"


class Map {

public:
    Map();
    void run();
    void prepare(std::vector<float>& p3D, std::vector<glm::mat4>& pose3d);

    void addPoint(Point* pt);
    std::vector<Point*> getPoints();

    void addFrame(KeyFrame* frame);
    std::vector<KeyFrame*> getFrames();

    void setCVFrame(cv::Mat frame);
    cv::Mat getCVFrame();

private:
    std::vector<KeyFrame*> frames;
    std::vector<Point*> points;
    cv::Mat cvFrame;

    std::mutex mutexFrames;
    std::mutex mutexPoints;
    std::mutex mutexCVFrame;

    GLFWwindow* window;
    Shader* ourShader; 
};

#endif 
