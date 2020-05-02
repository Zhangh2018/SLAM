#include "Init.h"

void threadBA(Optimizer& opt, Map& m, int iter);

void Init::triangulate(const cv::KeyPoint& keypoints1, const cv::KeyPoint& keypoints2, cv::Mat& P1, cv::Mat& P2, cv::Mat& points3d) {
    cv::Mat A(4,4, CV_32F);

    A.row(0) = keypoints1.pt.x * P1.row(2) - P1.row(0);
    A.row(1) = keypoints1.pt.y * P1.row(2) - P1.row(1);
    A.row(2) = keypoints2.pt.x * P2.row(2) - P2.row(0);
    A.row(3) = keypoints2.pt.y * P2.row(2) - P2.row(1);

    cv::Mat U, W, Vt;
    cv::SVD::compute(A, W, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    points3d = Vt.row(3).t();
    points3d = points3d.rowRange(0,3) / points3d.at<float>(3);
}

void Init::triangulate(const cv::Point2f& point1, const cv::Point2f& point2, cv::Mat& P1, cv::Mat& P2, cv::Mat& point3d) {
    cv::Mat A(4,4, CV_32F);

    A.row(0) = point1.x * P1.row(2) - P1.row(0);
    A.row(1) = point1.y * P1.row(2) - P1.row(1);
    A.row(2) = point2.x * P2.row(2) - P2.row(0);
    A.row(3) = point2.y * P2.row(2) - P2.row(1);

    cv::Mat U, W, Vt;
    cv::SVD::compute(A, W, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    point3d = Vt.row(3).t();
    point3d = point3d.rowRange(0,3) / point3d.at<float>(3);
}

// decompose Essential matrix to R, t; E = SR, S = [t]x
// TODO: consider another possible solution
void Init::decomposeE(const cv::Mat& E, cv::Mat& R, cv::Mat& t) {
    cv::Mat U, D, Vt;
    cv::SVD::compute(E, D, U, Vt);

    if (cv::determinant(Vt) < 0)
        Vt = -Vt;

    U.col(2).copyTo(t);
    t /= cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1) = -1;
    W.at<float>(1,0) = 1;
    W.at<float>(2,2) = 1;

    cv::Mat R1, R2;
    R1 = U * W * Vt;
    R2 = U * W.t() * Vt;
    if (R1.at<float>(0,0) + R1.at<float>(1,1) + R1.at<float>(2,2) > R2.at<float>(0,0) + R2.at<float>(1,1) + R2.at<float>(2,2))
        R = R1;
    else R = R2;
    if (cv::determinant(R) < 0)
        R = -R;
}

Init::Init(cv::VideoCapture& _cap, cv::VideoCapture& _cap1, const int _nfeatures, const int _thold, float _H, float _W) {
    cap = _cap;
    cap1 = _cap1;
    nfeatures = _nfeatures;
    thold = _thold;
    detector = cv::FastFeatureDetector::create(thold);
    desc = cv::ORB::create(nfeatures, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE, 31, thold);
    matcher = new cv::BFMatcher(cv::NORM_HAMMING, false);
    //matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    H = _H;
    W = _W;
    //matcher = new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));

}

void Init::extractKeyPoints(cv::Mat frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    detector->detect(frame, keypoints);
    //cv::goodFeaturesToTrack(frame, keypoints, 3000, 0.01f, 7);
    desc->compute(frame, keypoints, descriptors);
}

void Init::process() {
    Map m(H,W);
    std::thread t1(&Init::processFrames, this, std::ref(m));
    m.run();
}

struct Matches {
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    std::vector<cv::DMatch> mt;
    std::vector<uchar> mask;
};

Matches* filterMatches(std::vector<cv::KeyPoint>& keypoints1,
        std::vector<cv::KeyPoint>& keypoints2,
        cv::Mat& descriptors,
        std::vector<std::vector<cv::DMatch>>& matches, cv::Mat& F) {
    std::vector<cv::Point2f> pMatches1, pMatches2;
    std::vector<uchar> mask;
    std::vector<cv::DMatch> goodMatchesTemp, goodMatches;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (matches[i].size() < 2) break;
        const float ratio = 0.75;
        if (matches[i][0].distance < ratio * matches[i][1].distance) {
            goodMatchesTemp.push_back(matches[i][0]);
            // collecting points for compute fundamental matrix
            pMatches1.push_back(keypoints1[matches[i][0].queryIdx].pt);
            pMatches2.push_back(keypoints2[matches[i][0].trainIdx].pt);
        }
    }
    F = cv::findFundamentalMat(pMatches1, pMatches2, mask, cv::FM_RANSAC, 3.0f, 0.99f);
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i]) {
            goodMatches.push_back(goodMatchesTemp[i]);
        }
    }
    Matches* mt = new Matches{keypoints1, descriptors, goodMatches, mask};
    return mt;
}

void Init::processFrames(Map& m) {
    //instrisic params
    float f, cx, cy;
    f = 647.18f;
    cx = W / 2.0f; 
    cy = H / 2.0f;

    //intrinsic matrix
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = f;
    K.at<float>(1,1) = f;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;

    std::vector<cv::KeyPoint> keypoints1, keypoints2, keypoints3, keypoints4;
    cv::Mat descriptors1, descriptors2;
    std::vector<std::vector<cv::DMatch>> matches;

    std::vector<cv::Point2f> pMatches1, pMatches2;
    std::vector<cv::Point2f> goodPoints;
    std::vector<uchar> mask;
    cv::Mat frame;
    cv::Mat frame1;
    Matches *goodMatchesPrev, *goodMatchesCurrent, *goodMatches;

    Optimizer optimizer;
    bool TRACKING = false;
    while(1) {
        std::vector<KeyFrame*> frames = m.getFrames();
        std::vector<Point*> points = m.getPoints(); 
        if (frames.size() > 4) TRACKING = true;
        KeyFrame* keyframe = new KeyFrame(&K, frames.size());
        std::cout << "***** Frame: " << keyframe->id << " *****" << std::endl;
        cap >> frame;
        cap1 >> frame1;
        if (frame.empty() || frame1.empty()) break;
        cv::resize(frame, frame, cv::Size(), 0.99, 0.99);
        cv::resize(frame1, frame1, cv::Size(), 0.99, 0.99);

        keypoints1.clear();
        keypoints2.clear();
        matches.clear();
        extractKeyPoints(frame,  keypoints1, descriptors1);
        extractKeyPoints(frame1, keypoints2, descriptors2);
        matcher->knnMatch(descriptors1, descriptors2, matches, 2);
        std::cout << "keypoints1:                 " << keypoints1.size() << std::endl;
        std::cout << "keypoints2:                 " << keypoints2.size() << std::endl;
        std::cout << "matches:                    " << matches.size() << std::endl;

        cv::Mat F;
        goodMatchesCurrent = filterMatches(keypoints1, keypoints2,
                descriptors1, matches, F);

        if (goodMatchesPrev == nullptr) {
            goodMatchesPrev = goodMatchesCurrent;
            continue;
        }

        matches.clear();
        matcher->knnMatch(goodMatchesPrev->desc, goodMatchesCurrent->desc, matches, 2);
        delete(goodMatches);
        goodMatches = filterMatches(goodMatchesPrev->kp, goodMatchesCurrent->kp,
                goodMatchesPrev->desc, matches, F);

        F.convertTo(F, CV_32F);
        cv::Mat E = K.t()*F*K;
        cv::Mat R, t;
        //decomposeE(E, R, t);
        E.convertTo(E, CV_64F);
        std::vector<cv::Point2f> kp1, kp2;
        for (size_t i = 0; i < goodMatches->mask.size(); i++) {
            if (goodMatches->mask[i]) {
                kp1.push_back(goodMatchesPrev->kp[i].pt);
                kp2.push_back(goodMatchesCurrent->kp[i].pt);
            }
        }
        cv::recoverPose(E, kp1, kp2, K, R, t, cv::noArray());
        R.convertTo(R, CV_32F);
        t.convertTo(t, CV_32F);
        t /= 10;

        // Camera 1 Projection Matrix K[I|0]
        cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
        K.copyTo(P1.rowRange(0,3).colRange(0,3));
        cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

        // Camera 3 Projection Matrix K[R|t]
        cv::Mat P2(3,4,CV_32F, cv::Scalar(1));
        R.copyTo(P2.rowRange(0,3).colRange(0,3));
        t.copyTo(P2.rowRange(0,3).col(3));
        cv::Mat O2 = -R.t()*t;

        cv::Mat QQ = cv::Mat::eye(4,4,CV_32F);
        R.copyTo(QQ.rowRange(0,3).colRange(0,3));
        t.copyTo(QQ.rowRange(0,3).col(3));
        QQ = QQ.inv();
           
        cv::Mat K4x4 = cv::Mat::eye(4,4,CV_32F); 
        K.copyTo(K4x4.rowRange(0,3).colRange(0,3));

        if (frames.empty())
            keyframe->setPose(QQ); 
        else
            keyframe->setPose(frames.back()->getPose() * QQ);

        float th = keyframe->getPose().at<float>(2,3);
        std::cout << "Threshold:                  " << th << std::endl;
        //std::cout << cv::format(poseTest.back(), cv::Formatter::FMT_PYTHON) << std::endl;
        P2 = K*P2;

        std::vector<cv::DMatch> temp;
        cv::Mat s3DPoint;
        std::vector<cv::Point2f> rPoints;
        std::vector<int> rPointsIdx;

        // reproject map points
        for (size_t i = 0; i < points.size(); ++i) {
            std::vector<float> xyz = points[i]->getCoords();
            if (xyz[2] > th || cv::norm(xyz[2] - th) > 20) continue;
            cv::Mat kf = keyframe->getPose().inv();
            cv::Mat pt = (cv::Mat_<float>(4,1) << xyz[0], xyz[1], xyz[2], 1);
            pt = K4x4 * kf * pt; 

            // projecting homogeneous coords to euclidean
            pt /= pt.at<float>(3);
            //if (pt.at<float>(2) <= 0) continue;
            pt /= pt.at<float>(2);
            if (pt.at<float>(0) <= 0 || pt.at<float>(0) >= W ||
                pt.at<float>(1) <= 0 || pt.at<float>(1) >= H) continue;
            rPointsIdx.push_back(i);
            rPoints.push_back(cv::Point2f(pt.at<float>(0), pt.at<float>(1)));
        }

        /* print to frame already used mappoints as red ones
        for (auto& pt : rPoints) {
            cv::circle(frame, pt, 3, cv::Scalar(0,0,255));
        }
        */

        std::cout << "Points size: " << points.size() << std::endl << "rPoints size: " << rPoints.size() << std::endl;

        // mask is 1-d vector checking X'FX = 0
        int cnt = 0;
        for (size_t i = 0; i < goodMatches->kp.size(); ++i) {
            if (!rPoints.empty()) {
                bool flag = false;
                double minDistance = DBL_MAX;
                int mPoint = 0; 
                int mIdx = 0;
                for (size_t j = 0; j < rPointsIdx.size(); j++)  {
                    int k = rPointsIdx[j];
                    double distance = cv::norm(points[k]->getDesc(), goodMatches->desc.row(i), cv::NORM_HAMMING); 
                    float eDistance = euclideanDistance(rPoints[j], goodMatches->kp[i]);
                    if (distance <= 64 && eDistance <= 64) { 
                        if (distance < minDistance) {
                            minDistance = distance;
                            mPoint = k;
                            mIdx = j;
                        }
                        flag = true;
                    }
                }
                if (flag) {
                    points[mPoint]->addObservation(keyframe, keyframe->getKpSize());
                    keyframe->addKeypoint(goodMatches->kp[i], points[mPoint]->id);
                    cnt++;
                    rPoints.erase(rPoints.begin() + mIdx);
                    rPointsIdx.erase(rPointsIdx.begin() + mIdx);
                    continue;
                }
            }
            triangulate(goodMatchesPrev->kp[i], goodMatchesCurrent->kp[i], P1, P2, s3DPoint);
            // checking if coordinates goes to infinity
            if (!std::isfinite(s3DPoint.at<float>(0,0)) || !std::isfinite(s3DPoint.at<float>(1,0)) || !std::isfinite(s3DPoint.at<float>(2,0))) continue;

            // checking paralax
            cv::Mat norm1 = s3DPoint - O1;
            float dist1 = cv::norm(norm1);

            cv::Mat norm2 = s3DPoint - O2;
            float dist2 = cv::norm(norm2);

            float cosParallax = norm1.dot(norm2) / (dist1 * dist2);
            if (s3DPoint.at<float>(2,0) <= 0 && cosParallax < 0.99998) continue;

            cv::Mat s3DPoint2 = R*s3DPoint + t;
            if (s3DPoint2.at<float>(2) <= 0) continue;
            if (s3DPoint2.at<float>(2,0) <= 0 && cosParallax < 0.99998) continue;

            // coords for reprojection err
            float kp1x = goodMatchesPrev->kp[i].pt.x;
            float kp1y = goodMatchesPrev->kp[i].pt.y;
            float kp2x = goodMatchesCurrent->kp[i].pt.x;
            float kp2y = goodMatchesCurrent->kp[i].pt.y;

            // Check reprojection error in first image
            if (reprojErr(kp1x, kp1y, cx, cy, f, s3DPoint)) continue;

            // Check reprojection error in second image
            if (reprojErr(kp2x, kp2y, cx, cy, f, s3DPoint2)) continue;

            if (s3DPoint.at<float>(2) > th && cv::norm(s3DPoint.at<float>(2) - th) < 20) {
                cv::Vec3b intensity = frame.at<cv::Vec3b>(cv::Point(goodMatchesCurrent->kp[i].pt.x, goodMatchesCurrent->kp[i].pt.y));
                std::vector<float> color = {static_cast<float>(intensity.val[2]) / 255, static_cast<float>(intensity.val[1]) / 255, static_cast<float>(intensity.val[0]) / 255};
                Point* pt = new Point(m.getPointsSize(), s3DPoint.at<float>(0,0), s3DPoint.at<float>(1,0), s3DPoint.at<float>(2,0), goodMatches->desc.row(i), color);
                pt->addObservation(keyframe, keyframe->getKpSize());
                keyframe->addKeypoint(goodMatchesCurrent->kp[i], pt->id);
                m.addPoint(pt);
            }
        }
        std::cout << "Number of map points:       " << m.getPointsSize() << std::endl;
        std::cout << "Number of added map points: " << cnt << std::endl;

        //if (keyframe->getKpSize() < 100) keyframe->bad = true;
        m.addFrame(keyframe);

        std::cout << "good matches after tests:   " << goodMatches->mt.size() << std::endl;
        
        int iterations = 10;

        if (false && frames.size() % iterations == 0 && frames.size() > 0) {
            //std::thread t1(threadBA, std::ref(optimizer), std::ref(m), 5);
            //t1.join();
            optimizer.BundleAdjustment(m, 10, 0);
        }

        drawMatches(frame, goodMatchesPrev->kp, goodMatchesCurrent->kp, goodMatches->mt);
        delete(goodMatchesPrev);
        goodMatchesPrev = goodMatchesCurrent;
        m.setCVFrame(frame);

        //char c = (char)cv::waitKey(25);
        //if (c == 27) break;
    }
}

void Init::drawMatches(cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, std::vector<
cv::DMatch>& matches) {
    for (size_t i = 0; i < matches.size(); i++) {
        cv::Point2f coord1 = keypoints1[matches[i].queryIdx].pt;
        cv::Point2f coord2 = keypoints2[matches[i].trainIdx].pt;

        cv::circle(frame, coord1, 3, cv::Scalar(0,255,0));
        cv::circle(frame, coord2, 3, cv::Scalar(0,255,0));

        cv::line(frame, coord1, coord2, cv::Scalar(255,0,0));
    }
}

void Init::normalize(std::vector<cv::KeyPoint>& points, std::vector<cv::KeyPoint>& normalizedPoints, cv::Mat& T) {

    float mX = 0, mY = 0, d = 0;
    const size_t N = points.size();
    normalizedPoints.resize(N);

    for (size_t i = 0; i < N; ++i) {
        mX += points[i].pt.x;
        mY += points[i].pt.y;
    }

    mX /= N;
    mY /= N;

    for (size_t i = 0; i < N; ++i) {
        normalizedPoints[i].pt.x = points[i].pt.x - mX;
        normalizedPoints[i].pt.y = points[i].pt.y - mY;
        //d += std::sqrt(pow(normalizedPoints[i].pt.x, 2) + pow(normalizedPoints[i].pt.y, 2));
    }

    d = std::sqrt(2) * N / d;
     
    for (size_t i = 0; i < N; ++i) {
        normalizedPoints[i].pt.x *= d; 
        normalizedPoints[i].pt.y *= d;
    }

    T = cv::Mat::eye(3,3, CV_32F);
    T.at<float>(0,0) = 1;
    T.at<float>(1,1) = 1;
    T.at<float>(0,2) = -mX;
    T.at<float>(1,2) = -mY;
}

bool Init::reprojErr(float kpx, float kpy, float cx, float cy, float f, cv::Mat point) {
    float imx, imy;
    float invZ = 1.0 / point.at<float>(2,0);
    imx = f*point.at<float>(0,0)*invZ + cx;
    imy = f*point.at<float>(1,0)*invZ + cy;
    
    float squareError = (imx-kpx)*(imx-kpx)+(imy-kpy)*(imy-kpy);

    if(squareError > 4) return true;
    else return false;
}

float Init::euclideanDistance(cv::Point2f pt1, cv::KeyPoint pt2) {
    return sqrt(pow(pt1.x - pt2.pt.x, 2) + pow(pt1.y - pt2.pt.y, 2));
}

void threadBA(Optimizer& opt, Map& m, int iter) {
    opt.BundleAdjustment(m, iter, 0);
}
