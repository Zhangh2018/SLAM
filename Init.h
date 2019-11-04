#ifndef INIT_H
#define INIT_H
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "map.h"


class Init {
public:
    Init(cv::VideoCapture& cap, const int nfeatures, const int thold);
    void processFrames();
private:
    void extractKeyPoints(cv::Mat frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    void drawMatches(cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches);
    void normalize(const std::vector<cv::Point2f>& keypoints, std::vector<cv::Point2f>& normalizedPoints, cv::Mat& T);
    void triangulate(const cv::KeyPoint keypoints1, cv::KeyPoint keypoints2, cv::Mat& P1, cv::Mat& P2, cv::Mat& points3d);
    void decomposeE(const cv::Mat& E, cv::Mat& R, cv::Mat& t);
    // video wrapper
    cv::VideoCapture cap;

    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::FeatureDetector> desc;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    int thold;
    int nfeatures;

};

#endif // INIT_H
