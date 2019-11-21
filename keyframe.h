#ifndef KEYFRAME_H
#define KEYFRAME_H
#include <opencv2/core/core.hpp>
#include <glm/glm.hpp>
#include <iostream> 
#include <vector>

class KeyFrame;

class Point {
public:
    Point(cv::Mat* K, float x, float y, float z, cv::DMatch desc, KeyFrame* frame);
    cv::Mat* K;
    std::vector<float> xyz;
    std::vector<KeyFrame*> frames;
    cv::DMatch desc;
};

// make getters and setters
class KeyFrame {

public:
    KeyFrame();
    
    glm::mat4 pose;
    std::vector<Point*> kp;
    std::vector<int> desc;

};


#endif
