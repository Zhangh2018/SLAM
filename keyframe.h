#ifndef KEYFRAME_H
#define KEYFRAME_H
#include <opencv2/core/core.hpp>
#include <opencv2/flann.hpp>
#include <glm/glm.hpp>
#include <iostream> 
#include <vector>
#include <map>
#include <mutex>

class KeyFrame;

class Point {
public:
    Point(int id, float x, float y, float z);
    int id;

public:
    void setCoords(float x, float y, float z);
    std::vector<float> getCoords();
    void addObservation(Keyframe* kf, int idx);
    std::map<KeyFrame*, idx> getObservations();

private:
    std::vector<float> xyz;
    std::map<KeyFrame*, int> obs;
    std::mutex mutexPoint;
    std::mutex mutexObservation;
};

// make getters and setters
class KeyFrame {

public:
    KeyFrame(cv::Mat* K, int id);
    
    void setPose(cv::Mat pose);
    cv::Mat getPose();
    void addKeypoint(cv::KeyPoint kp, int desc);
    cv::KeyPoint getKeypoint(int idx);
    int getKpSize();

public:
    cv::Mat* K;
    int id;
    bool bad = false;

private:
    cv::Mat pose;
    std::vector<cv::KeyPoint> kp;
    std::vector<int> desc;

    std::mutex mutexPose;
    std::mutex mutexKeypoint;
};


#endif
