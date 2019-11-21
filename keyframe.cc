#include "keyframe.h"

Point::Point(cv::Mat* _K, float x, float y, float z, cv::DMatch _desc, KeyFrame* frame) {
    frames.push_back(frame);
    K = _K;
    xyz.push_back(x);
    xyz.push_back(y);
    xyz.push_back(z);
    desc = _desc;
}

KeyFrame::KeyFrame() {
}

