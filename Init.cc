#include "Init.h"

std::vector<float> frustum(float poseX, float poseY, float poseZ);
bool fromCV2GLM(const cv::Mat& cvmat, glm::mat4* glmmat);

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
    std::vector<float> p3D;
    std::vector<glm::mat4> pose;
    Map m; // map object

    //instrisic params
    float f, cx, cy;
    f = 1000.0f;
    cx = 1280.0f * 0.75f / 2.0f;
    cy = 720.0f * 0.75f / 2.0f;

    //intrinsic matrix
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = f;
    K.at<float>(1,1) = f;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;

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

    pose.push_back(glm::mat4(1.0f));

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

        // for some reason working better without normalizarion
        //cv::Mat T1, T2;
        //normalize(keypoints1, keypoints1, T1);
        //normalize(keypoints2, keypoints2, T2);

        goodMatches.clear();
        pMatches1.clear(); pMatches2.clear();
        // Lowe's ratio test
		for (int i = 0; i < matches.size(); ++i) {
			if (matches[i].size() < 2) break;
		   	const float ratio = 0.75;
		   	if (matches[i][0].distance < ratio * matches[i][1].distance) {
		    	goodMatches.push_back(matches[i][0]);
		    	// collecting points for compute fundamental matrix
                pMatches1.push_back(keypoints1[matches[i][0].queryIdx].pt);
                pMatches2.push_back(keypoints2[matches[i][0].trainIdx].pt);
		   	}
        }
        std::cout << "good matches: " << goodMatches.size() << std::endl;
        
        mask.clear();
        cv::Mat F = cv::findFundamentalMat(pMatches1, pMatches2, mask, cv::FM_RANSAC, 3.0f, 0.99f);
        F.convertTo(F, CV_32F);
        //F = T2.t() * F * T1;
        cv::Mat E = K.t()*F*K;
        cv::Mat R, t;
        //decomposeE(E, R, t);
		E.convertTo(E, CV_64F);
        cv::recoverPose(E, pMatches1, pMatches2, K, R, t, cv::noArray());
        R.convertTo(R, CV_32F);
        t.convertTo(t, CV_32F);
		t /= 10;
		//R = R.t();

        // Camera 1 Projection Matrix K[I|0]
        cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
        K.copyTo(P1.rowRange(0,3).colRange(0,3));
        cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

        // Camera 2 Projection Matrix K[R|t]
        cv::Mat P2(3,4,CV_32F, cv::Scalar(1));
        R.copyTo(P2.rowRange(0,3).colRange(0,3));
        t.copyTo(P2.rowRange(0,3).col(3));
        cv::Mat O2 = -R.t()*t;

        cv::Mat QQ = cv::Mat::eye(4,4,CV_32F);
        R.copyTo(QQ.rowRange(0,3).colRange(0,3));
        t.copyTo(QQ.rowRange(0,3).col(3));
        QQ = QQ.t();
		glm::mat4 ttemp;
		ttemp[0][0] = QQ.at<float>(0,0);
		ttemp[0][1] = QQ.at<float>(0,1);
		ttemp[0][2] = QQ.at<float>(0,2);
		ttemp[0][3] = QQ.at<float>(0,3);
		ttemp[1][0] = QQ.at<float>(1,0);
		ttemp[1][1] = QQ.at<float>(1,1);
		ttemp[1][2] = QQ.at<float>(1,2);
		ttemp[1][3] = QQ.at<float>(1,3);
		ttemp[2][0] = QQ.at<float>(2,0);
		ttemp[2][1] = QQ.at<float>(2,1);
		ttemp[2][2] = QQ.at<float>(2,2);
		ttemp[2][3] = QQ.at<float>(2,3);
		ttemp[3][0] = QQ.at<float>(3,0);
		ttemp[3][1] = QQ.at<float>(3,1);
		ttemp[3][2] = QQ.at<float>(3,2);
		ttemp[3][3] = QQ.at<float>(3,3);
        pose.push_back(pose.back() * ttemp);
        float th = pose.back()[3][2];
        //std::cout << cv::format(poseTest.back(), cv::Formatter::FMT_PYTHON) << std::endl;
        P2 = K*P2;

        std::vector<cv::DMatch> temp;
        cv::Mat s3DPoint;
        // mask is 1-d vector checking X'FX = 0
        for (int i = 0; i < mask.size(); ++i) {
            if (mask[i]) {
                temp.push_back(goodMatches[i]);
                triangulate(keypoints1[goodMatches[i].queryIdx], keypoints2[goodMatches[i].trainIdx], P1, P2, s3DPoint);
                if (!std::isfinite(s3DPoint.at<float>(0,0)) || !std::isfinite(s3DPoint.at<float>(1,0)) || !std::isfinite(s3DPoint.at<float>(2,0))) continue;
                // check paralax
                cv::Mat norm1 = s3DPoint - O1;
                float dist1 = cv::norm(norm1);

                cv::Mat norm2 = s3DPoint - O2;
                float dist2 = cv::norm(norm2);

                float cosParallax = norm1.dot(norm2) / (dist1 * dist2);

                if (s3DPoint.at<float>(2,0) <= 0 && cosParallax < 0.99998) continue;

                cv::Mat s3DPoint2 = R*s3DPoint + t;

                if (s3DPoint2.at<float>(2,0) <= 0 && cosParallax < 0.99998) continue;

                if (-s3DPoint.at<float>(2,0) < th) {
                    // Redo to use more convenient way to store points
                    p3D.push_back(s3DPoint.at<float>(0,0));
                    p3D.push_back(-s3DPoint.at<float>(1,0));
                    p3D.push_back(-s3DPoint.at<float>(2,0));
                }
            }
        }
        goodMatches = temp;
        std::cout << "good matches after RANSAC: " << goodMatches.size() << std::endl;
        drawMatches(frame, keypoints1, keypoints2, goodMatches);
        cv::imshow("Keypoints", frame);
        m.run(p3D, pose);
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

void Init::normalize(std::vector<cv::KeyPoint>& points, std::vector<cv::KeyPoint>& normalizedPoints, cv::Mat& T) {

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
        normalizedPoints[i].pt.x = points[i].pt.x - mX;
        normalizedPoints[i].pt.y = points[i].pt.y - mY;
        //d += std::sqrt(pow(normalizedPoints[i].pt.x, 2) + pow(normalizedPoints[i].pt.y, 2));
    }

    //d = std::sqrt(2) * N / d;
    /* 
    for (int i = 0; i < N; ++i) {
        normalizedPoints[i].pt.x *= d; 
        normalizedPoints[i].pt.y *= d;
    }
    */

    T = cv::Mat::eye(3,3, CV_32F);
    T.at<float>(0,0) = 1;
    T.at<float>(1,1) = 1;
    T.at<float>(0,2) = -mX;
    T.at<float>(1,2) = -mY;
}

std::vector<float> frustum(float poseX, float poseY, float poseZ) {

    std::vector<float> frustumModel{

         -0.05f + poseX, 0.05f + poseY, 0.05f + poseZ,
         0.05f + poseX, 0.05f + poseY, 0.05f + poseZ,
         0.0f + poseX, 0.0f + poseY, 0.0f + poseZ,

         0.0f  + poseX, 0.0f   + poseY, 0.0f + poseZ,
         0.05f + poseX, 0.0f   + poseY, 0.05f + poseZ,
         0.05f + poseX, 0.05f  + poseY, 0.05f + poseZ,

         0.0f   + poseX, 0.0f  + poseY, 0.0f + poseZ,
         -0.05f + poseX, 0.0f  + poseY, 0.05f + poseZ,
         -0.05f + poseX, 0.05f + poseY, 0.05f + poseZ,

         0.0f   + poseX, 0.0f + poseY, 0.0f + poseZ,
         -0.05f + poseX, 0.0f + poseY, 0.05f + poseZ,
         0.05f  + poseX, 0.0f + poseY, 0.05f + poseZ,

         -0.05f + poseX, 0.05f + poseY, 0.05f + poseZ,
         0.05f  + poseX, 0.0f + poseY, 0.05f + poseZ,
         0.05f  + poseX, 0.0f + poseY, 0.05f + poseZ,

         -0.05f + poseX, 0.05f + poseY, 0.05f + poseZ,
         0.05f  + poseX, 0.05f + poseY, 0.05f + poseZ,
         0.05f  + poseX, 0.0f + poseY, 0.05f + poseZ
    };
      
    return frustumModel;
}


glm::mat4 fromCV2GLM(const cv::Mat& cvmat) {
   	glm::mat4 temp;
    temp[0][0] = cvmat.at<float>(0,0);
    temp[0][1] = cvmat.at<float>(1,0);
    temp[0][2] = cvmat.at<float>(2,0);
    temp[0][3] = cvmat.at<float>(3,0);
    temp[1][0] = cvmat.at<float>(0,1);
    temp[1][1] = cvmat.at<float>(1,1);
    temp[1][2] = cvmat.at<float>(2,1);
    temp[1][3] = cvmat.at<float>(3,1);
    temp[2][0] = cvmat.at<float>(0,2);
    temp[2][1] = cvmat.at<float>(1,2);
    temp[2][2] = cvmat.at<float>(2,2);
    temp[2][3] = cvmat.at<float>(3,2);
    temp[3][0] = cvmat.at<float>(0,3);
    temp[3][1] = cvmat.at<float>(1,3);
    temp[3][2] = cvmat.at<float>(2,3);
    temp[3][3] = cvmat.at<float>(3,3);

    return temp;
}

bool fromGLM2CV(const glm::mat4& glmmat, cv::Mat* cvmat) {
   	if (cvmat->cols != 4 || cvmat->rows != 4) {
   		(*cvmat) = cv::Mat(4, 4, CV_32F);
   	}
   	memcpy(cvmat->data, glm::value_ptr(glmmat), 16 * sizeof(float));
   	*cvmat = cvmat->t();
   	return true;
}
