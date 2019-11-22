#include "optimizer.h"


Eigen::Matrix<double,3,1> toVector3d(std::vector<float>& vec);
cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
g2o::SE3Quat toSE3Quat(const cv::Mat &cvt);
cv::Mat toCvMat(const g2o::SE3Quat &SE3);
std::vector<float> toStdVector(const Eigen::Matrix<double,3,1> &m);


void Optimizer::BundleAdjustment(Map& m, int iter) {
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    for (int i = 0; i < m.frames.size(); i++) {
        KeyFrame* kf = m.frames[i];
        if (kf->bad) continue;
        g2o::VertexSE3Expmap* SE3 = new g2o::VertexSE3Expmap();
        SE3->setEstimate(toSE3Quat(kf->pose));
        SE3->setId(kf->id);
        SE3->setFixed(kf->id == 0);
        optimizer.addVertex(SE3);
    }

    const int maxId = m.frames.size();
    const float thHuber = sqrt(5.991);

    for (int i = 0; i < m.points.size(); i++) {
        Point* pt  = m.points[i];
        g2o::VertexSBAPointXYZ* Point = new g2o::VertexSBAPointXYZ();
        Point->setEstimate(toVector3d(pt->xyz));
        int id = pt->id + maxId + 1;
        Point->setId(id);
        Point->setMarginalized(true);
        optimizer.addVertex(Point);

        for (auto& it: pt->obs) {
            KeyFrame* kf = it.first;
            if (kf->bad) continue;
            Eigen::Matrix<double,2,1> obs;
            cv::KeyPoint kp = kf->kp[it.second];
            obs << kp.pt.x, kp.pt.y;

            g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(kf->id)));
            e->setMeasurement(obs);
            e->setInformation(Eigen::Matrix2d::Identity());

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(thHuber);

            e->fx = kf->K->at<float>(0,0);
            e->fy = kf->K->at<float>(1,1);
            e->cx = kf->K->at<float>(2,0);
            e->cx = kf->K->at<float>(2,1);

            optimizer.addEdge(e);
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(iter);

    for (int i = 0; i < m.frames.size(); i++) {
        KeyFrame* kf = m.frames[i];
        if (kf->bad) continue;
        g2o::VertexSE3Expmap* SE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(kf->id));
        g2o::SE3Quat SE3quat = SE3->estimate();
        kf->pose = toCvMat(SE3quat);
    }

    for (int i = 0; i < m.points.size(); i++) {
        Point* pt = m.points[i];
        g2o::VertexSBAPointXYZ* Point = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pt->id + maxId + 1));
        pt->xyz = toStdVector(Point->estimate());
    }
}


Eigen::Matrix<double,3,1> toVector3d(std::vector<float>& vec) {
    Eigen::Matrix<double,3,1> v;
    v << vec[0], vec[1], vec[2]; 
    return v;
}

std::vector<float> toStdVector(const Eigen::Matrix<double,3,1> &m) {
    std::vector<float> vec;
    for (int i = 0; i < 3; i++) {
        vec.push_back(m(i));
    }

    return vec;
}


cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m) {
    cv::Mat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++)
            cvMat.at<float>(i)=m(i);

    return cvMat.clone();
}

cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}


g2o::SE3Quat toSE3Quat(const cv::Mat &cvT) {
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
         cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
         cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

    return g2o::SE3Quat(R,t);
}
