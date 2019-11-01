#include "Init.h"

Init::Init(cv::VideoCapture& _cap, const int _nfeatures, const int _thold) {
    cap = _cap;
    nfeatures = _nfeatures;
    thold = _thold;
    detector = cv::FastFeatureDetector::create(thold);
    desc = cv::ORB::create(nfeatures, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE, 31, 20);
    matcher = new cv::BFMatcher(cv::NORM_HAMMING, false);
    //matcher = new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));

}

void Init::extractKeyPoints(cv::Mat frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    detector->detect(frame, keypoints);
    desc->compute(frame, keypoints, descriptors);
}

void Init::processFrames() {
    cv::Mat frame;
    cap >> frame;
    cv::resize(frame, frame, cv::Size(), 0.75, 0.75);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    std::vector<std::vector<cv::DMatch>> matches;

    std::vector<cv::DMatch> goodMatches;
    std::vector<cv::Point2f> pMatches1, pMatches2;
    cv::Mat T12, T21;
    std::vector<uchar> mask;

    extractKeyPoints(frame, keypoints1, descriptors1);
    while(1) {
        cap >> frame;
        if (frame.empty()) break;
        cv::resize(frame, frame, cv::Size(), 0.75, 0.75);
 
        keypoints2.clear();
        matches.clear();
        extractKeyPoints(frame, keypoints2, descriptors2);
        matcher->knnMatch(descriptors1, descriptors2, matches, 2);
        std::cout << "keypoints1: " << keypoints1.size() << " keypoints2: " << keypoints2.size() << " matches: " << matches.size() << std::endl;

        goodMatches.clear();
        pMatches1.clear(); pMatches2.clear();
		for (int i = 0; i < matches.size(); ++i) {
			if (matches[i].size() < 2) break;
		   	const float ratio = 0.7;
		   	if (matches[i][0].distance < ratio * matches[i][1].distance) {
		    	goodMatches.push_back(matches[i][0]);
                pMatches1.push_back(keypoints1[matches[i][0].queryIdx].pt);
                pMatches2.push_back(keypoints2[matches[i][0].trainIdx].pt);
		   	}
        }
        std::cout << "good matches: " << goodMatches.size() << std::endl;

        mask.clear();
        cv::Mat F = cv::findFundamentalMat(pMatches1, pMatches2, mask, cv::FM_RANSAC, 3.0f, 0.99f);
        std::vector<cv::DMatch> temp;
        for (int i = 0; i < mask.size(); ++i) {
            if (mask[i]) {
                temp.push_back(goodMatches[i]);
            }
        }
        goodMatches = temp;
        std::cout << "good matches after RANSAC: " << goodMatches.size() << std::endl;
        drawMatches(frame, keypoints1, keypoints2, goodMatches);
        cv::imshow("Keypoints", frame);
        keypoints1 = keypoints2;
        descriptors1 = descriptors2;
        char c = (char)cv::waitKey(25);
        if (c == 27) break;
    }
}

void Init::drawMatches(cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2, std::vector<
cv::DMatch>& matches) {
    for (int i = 0; i < matches.size(); i++) {
        cv::Point2f coord1 = keypoints_1[matches[i].queryIdx].pt;
        cv::Point2f coord2 = keypoints_2[matches[i].trainIdx].pt;

        cv::circle(frame, coord1, 3, cv::Scalar(0,255,0));
        cv::circle(frame, coord2, 3, cv::Scalar(0,255,0));

        cv::line(frame, coord1, coord2, cv::Scalar(255,0,0));
    }
}


void Init::normalize(const std::vector<cv::KeyPoint>& points, std::vector<cv::Point2f>& normalizedPoints, cv::Mat& T) {
    
    float mX = 0, mY = 0, d = 0;
    const int N = points.size();
    normalizedPoints.resize(N);

    for (int i = 0; i < N; ++i) {
        mX += points[i].pt.x;
        mY += points[i].pt.y;
    }

    mX /= N;
    mY /= N;

    for (int i = 0; i < N; ++i) {
        normalizedPoints[i].x = points[i].pt.x - mX;
        normalizedPoints[i].y = points[i].pt.y - mY;
        d += std::sqrt(pow(normalizedPoints[i].x, 2) + pow(normalizedPoints[i].y, 2));
    }

    d = std::sqrt(2) * N / d;
    
    for (int i = 0; i < N; ++i) {
        normalizedPoints[i].x *= d; 
        normalizedPoints[i].y *= d;
    }

    T = cv::Mat::eye(3,3, CV_32F);
    T.at<float>(0,0) = d;
    T.at<float>(1,1) = d;
    T.at<float>(0,2) = -mX*d;
    T.at<float>(1,2) = -mY*d;
}
