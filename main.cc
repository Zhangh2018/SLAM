#include <iostream>
#include "Init.h"

int main() {
    cv::VideoCapture cap("./../test_video.mp4");
    
    Init init(cap, 10000, 25);
    init.processFrames();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
