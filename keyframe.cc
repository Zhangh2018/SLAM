#include "keyframe.h"


// Point

Point::Point(int _id, float x, float y, float z, cv::Mat _desc, std::vector<float> _color) {
    xyz.push_back(x);
    xyz.push_back(y);
    xyz.push_back(z);
    id = _id;
    desc = _desc;
    color = _color;
}

void Point::setCoords(float x, float y, float z) {
    std::unique_lock<std::mutex> lock(mutexPoint);
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;
}

void Point::setCoords(std::vector<float> _xyz) {
    std::unique_lock<std::mutex> lock(mutexPoint);
    xyz = _xyz;
}

std::vector<float> Point::getCoords() {
    std::unique_lock<std::mutex> lock(mutexPoint);
    return xyz;
}

void Point::addObservation(KeyFrame* kf, int idx) {
    std::unique_lock<std::mutex> lock(mutexObservation);
    obs[kf] = idx;
}

std::map<KeyFrame*, int> Point::getObservations() {
    std::unique_lock<std::mutex> lock(mutexObservation);
    return obs;
}

cv::Mat Point::getDesc() {
    return desc;
}

void Point::setColor(std::vector<float> _color) {
    color = _color;
}

std::vector<float> Point::getColor() {
    return color;
}

// KeyFrame

KeyFrame::KeyFrame(cv::Mat* _K, int _id) {
    K = _K;
    id = _id;
}

void KeyFrame::setPose(cv::Mat _pose) {
    std::unique_lock<std::mutex> lock(mutexPose);
    pose = _pose;
}

cv::Mat KeyFrame::getPose() {
    std::unique_lock<std::mutex> lock(mutexPose);
    return pose.clone();
}

void KeyFrame::addKeypoint(cv::KeyPoint keypoint, int ptsId) {
    std::unique_lock<std::mutex> lock(mutexKeypoint);
    kp.push_back(keypoint);
    pointsId.push_back(ptsId);
}

cv::KeyPoint KeyFrame::getKeypoint(int idx) {
     std::unique_lock<std::mutex> lock(mutexKeypoint);
     return kp[idx];
}

int KeyFrame::getKpSize() {
    std::unique_lock<std::mutex> lock(mutexKeypoint);
    return kp.size();
}

cv::Mat KeyFrame::getRotation() {
    std::unique_lock<std::mutex> lock(mutexPose);
    cv::Mat ret;
    pose.rowRange(0,3).colRange(0,3).copyTo(ret);
    return ret;
}

cv::Mat KeyFrame::getTranslation() {
    std::unique_lock<std::mutex> lock(mutexPose);
    cv::Mat ret;
    pose.rowRange(0,3).col(3).copyTo(ret);
    return ret;
}

cv::Mat KeyFrame::getRt() {
    std::unique_lock<std::mutex> lock(mutexPose);
    cv::Mat ret;
    pose.rowRange(0,3).colRange(0,4).copyTo(ret);
    return ret;
}

std::vector<int> KeyFrame::getPointsId() {
    std::unique_lock<std::mutex> lock(mutexKeypoint);
    return pointsId;
}
