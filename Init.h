#ifndef INIT_H
#define INIT_H
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thread>
#include "map.h"
#include "optimizer.h"


class Init {
public:
    Init(cv::VideoCapture& cap, const int nfeatures, const int thold, float H, float W);
    void process();
    void processFrames(Map& m);
    float H;
    float W;
private:
    void extractKeyPoints(cv::Mat frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    void drawMatches(cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches);
    void normalize(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::KeyPoint>& normalizedPoints, cv::Mat& T);
    void triangulate(const cv::KeyPoint& keypoints1, const cv::KeyPoint& keypoints2, cv::Mat& P1, cv::Mat& P2, cv::Mat& points3d);
    void triangulate(const cv::Point2f& point1, const cv::Point2f& point2, cv::Mat& P1, cv::Mat& P2, cv::Mat& point3d);
    void decomposeE(const cv::Mat& E, cv::Mat& R, cv::Mat& t);
    float euclideanDistance(cv::Point2f pt1, cv::KeyPoint pt2);

    bool reprojErr(float kpx, float kpy, float cx, float cy, float f, cv::Mat point);
    // video wrapper
    cv::VideoCapture cap;

    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::Feature2D> desc;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    int thold;
    int nfeatures;

};

#endif // INIT_H
