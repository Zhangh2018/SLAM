#include <iostream>
#include "Init.h"
#include <thread> 

int main(int argc, char* argv[]) {
    cv::VideoCapture cap("./../stereo/I1_%06d.png");
    cv::VideoCapture cap1("./../stereo/I2_%06d.png");
    int H = 0, W = 0;
    H = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    W = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    
    Init init(cap, cap1, 10000, 30, H, W);
    init.process();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
