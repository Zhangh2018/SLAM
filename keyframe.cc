#include "keyframe.h"

Point::Point(int _id, float x, float y, float z) {
    xyz.push_back(x);
    xyz.push_back(y);
    xyz.push_back(z);
    id = _id;
}

KeyFrame::KeyFrame(cv::Mat* _K, int _id) {
    K = _K;
    id = _id;
}

