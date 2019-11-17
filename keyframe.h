#ifndef KEYFRAME_H
#define KEYFRAME_H
#include <opencv2/core/core.hpp>
#include <glm/glm.hpp>
#include <iostream> 
#include <vector>

class KeyFrame;

class Point {
public:
    Point(cv::Mat* K, float x, float y, float z, KeyFrame* frame);
    cv::Mat* K;
    std::vector<float> xyz;
    std::vector<KeyFrame*> frames;

};

// make getters and setters
class KeyFrame {

public:
    KeyFrame();
    
    glm::mat4 pose;
    std::vector<Point*> kp;
    std::vector<cv::DMatch*> desc;

};


#endif
