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
    Point(int id, float x, float y, float z, cv::Mat desc,
          std::vector<float> color);
    int id;

  public:
    void setCoords(float x, float y, float z);
    void setCoords(std::vector<float> xyz);
    std::vector<float> getCoords();
    void addObservation(KeyFrame* kf, int idx);
    std::map<KeyFrame*, int> getObservations();
    cv::Mat getDesc();
    void setColor(std::vector<float> color);
    std::vector<float> getColor();

  private:
    std::vector<float> xyz;
    std::vector<float> color;
    std::map<KeyFrame*, int> obs;
    cv::Mat desc;
    std::mutex mutexPoint;
    std::mutex mutexObservation;
};

// make getters and setters
class KeyFrame {
  public:
    KeyFrame(cv::Mat* K, int id);

    void setPose(cv::Mat pose);
    cv::Mat getPose();
    void addKeypoint(cv::KeyPoint kp, int ptsId);
    cv::KeyPoint getKeypoint(int idx);
    int getKpSize();
    cv::Mat getRotation();
    cv::Mat getTranslation();
    cv::Mat getRt();
    std::vector<int> getPointsId();

  public:
    cv::Mat* K;
    int id;
    bool bad = false;

  private:
    cv::Mat pose;
    std::vector<cv::KeyPoint> kp;
    std::vector<int> pointsId;

    std::mutex mutexPose;
    std::mutex mutexKeypoint;
};

#endif
