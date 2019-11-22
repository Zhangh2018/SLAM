#include <iostream>
#include "Init.h"
#include <thread> 

int main() {
    cv::VideoCapture cap("./../test_video.mp4");
    
    Init init(cap, 1000, 25);
    init.processFrames();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
