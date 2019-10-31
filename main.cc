#include <iostream>
#include "Init.h"

int main() {
    cv::VideoCapture cap("./../test_video.mp4");
    
    Init init(cap, 5000, 25);
    init.ProcessFrames();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
