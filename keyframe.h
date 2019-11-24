#ifndef KEYFRAME_H
#define KEYFRAME_H
#include <opencv2/core/core.hpp>
#include <opencv2/flann.hpp>
#include <glm/glm.hpp>
#include <iostream> 
#include <vector>
#include <map>

class KeyFrame;

class Point {
public:
    Point(int id, float x, float y, float z);
    int id;
    std::vector<float> xyz;
    std::map<KeyFrame*, int> obs;
};

// make getters and setters
class KeyFrame {

public:
    KeyFrame(cv::Mat* K, int id);
    
    cv::Mat* K;
    int id;
    cv::Mat pose;
    std::vector<cv::KeyPoint> kp;
    std::vector<int> desc;

    bool bad = false;
};


#endif
