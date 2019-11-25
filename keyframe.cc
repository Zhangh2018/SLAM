#include "keyframe.h"


// Point

Point::Point(int _id, float x, float y, float z) {
    xyz.push_back(x);
    xyz.push_back(y);
    xyz.push_back(z);
    id = _id;
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
    return pose;
}

void KeyFrame::addKeypoint(cv::KeyPoint keypoint, int idx) {
    std::unique_lock<std::mutex> lock(mutexKeypoint);
    kp.push_back(keypoint);
    desc.push_back(idx);
}

cv::KeyPoint KeyFrame::getKeypoint(int idx) {
     std::unique_lock<std::mutex> lock(mutexKeypoint);
     return kp[idx];
}

int KeyFrame::getKpSize() {
    std::unique_lock<std::mutex> lock(mutexKeypoint);
    return kp.size();
}
