#include <iostream>
#include "Init.h"
#include <thread> 

int main(int argc, char* argv[]) {
    cv::VideoCapture cap("./../test_kitti984.mp4");
    int H = 0, W = 0;
    H = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    W = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    
    Init init(cap, 3000, 20, H, W);
    init.process();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
