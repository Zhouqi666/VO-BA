
#include "myslam.h"
#include "camera.h"
#include "frame.h"
#include "mappoint.h"
#include "map.h"
#include <time.h>



typedef Eigen::Matrix<double, 6, 1> Vector6d;

inline void GetImgandEqualHist(string &str_left, string &str_right, cv::Mat &left, cv::Mat &right);

inline void GetPoseLeftandRight(const double T_c_w_ceres[6], const double tr,Vector6d &pose3);

void ResultEstimation(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pose_left, const string &fp_truth);

void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses2);

void DenseMatch(const cv::Mat &img_left,
                const cv::Mat &img_right,
                const myslam::Map::Ptr map1,
                mrpt::maps::CColouredPointsMap &point4d);

int main(int argc, char** argv)
{
    vector<string> str_left, str_right, str_disparity;
    cv::Mat img_left, img_right, img_left2, img_right2;
    int img_row = 1241;
    int img_col = 376;
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pose_left;//final trajectory

    ifstream infile_left("/home/zero/NewDisk/sequences/00/image_0/imagelist_left.txt");
    ifstream infile_right("/home/zero/NewDisk/sequences/00/image_1/imagelist_right.txt");
    ifstream infile_disparity("/home/zero/NewDisk/sequences/00/disparityGC/imagelist_disparity.txt");
    while (!infile_left.eof())
    {
        string str_temp;
        getline(infile_left, str_temp);
        str_left.push_back(str_temp);

        getline(infile_right, str_temp);
        str_right.push_back(str_temp);

        getline(infile_disparity, str_temp);
        str_disparity.push_back(str_temp);
    }

    ofstream fp1("/home/zero/Desktop/estimate_trajectory/vo_ba00.txt", ios::trunc);
    string fp_truth = "/home/zero/Desktop/VO_BA/ground_truth/00.txt";
    ifstream ifs(fp_truth);
    int ba_window = 5;
    double time_sum = 0; //cost time
    bool initial = false;
    double reproject_error = 1.0;
    double tr = 537.165718864;
//    double tr = 537.150588;
//    double tr = 537.150653;
//      kitti 00-02
    cv::Mat camera_mat = (cv::Mat_<double>(3, 3) << 718.856, 0,       607.1928,
                          0,       718.856, 185.2157,
                          0,       0,       1        );
//    kitti 03
//            cv::Mat camera_mat = (cv::Mat_<double>(3, 3) << 721.5377, 0,       609.5593,
//                                  0,       721.5377, 172.854,
//                                  0,       0,       1        );
//    kitti 04-10
//            cv::Mat camera_mat = (cv::Mat_<double>(3, 3) << 707.0912, 0,       601.8873,
//                                  0,       707.0912, 183.1104,
//                                  0,       0,       1        );
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32FC1);//distCoeff vector in opencv

    //camera parameter
    myslam::Camera* kitti = new myslam::Camera(camera_mat, distCoeffs, tr, img_row, img_col);
    myslam::Map::Ptr map_ptr(new myslam::Map(kitti, ba_window, reproject_error));
    double first_initial_left[6] = {0, 0, 0, 0, 0, 0};
    Vector6d pose_temp;
    pose_temp << 0, 0, 0, 0, 0, 0;
    pose_left.push_back(pose_temp);

//////MRPT Visualization
//    mrpt::gui::CDisplayWindow3D window("Points Cloud Map", 1200, 600);
//    mrpt::opengl::COpenGLScenePtr scene;
//    mrpt::maps::CColouredPointsMap pointsAll;
//    scene = window.get3DSceneAndLock();

//    mrpt::opengl::CPointCloudColouredPtr kinectp = mrpt::opengl::CPointCloudColoured::Create();
//    scene->insert( kinectp );

//    mrpt::opengl::COpenGLViewportPtr vi= scene->createViewport("image");
//    vi->setViewportPosition(0.751, 0, 0.497, 0.15);

//    mrpt::opengl::COpenGLViewportPtr viTop= scene->createViewport("top");
//    viTop->setViewportPosition(0.751, 0.15, 0.2485, 0.2485);
//    viTop->setCloneView("main");
//    viTop->setTransparent(false);
//    viTop->getCamera().setAzimuthDegrees(-90);
//    viTop->getCamera().setElevationDegrees(0);
//    viTop->getCamera().setZoomDistance(350);
//    window.setPos(0,0);
//    window.unlockAccess3DScene();
//    window.repaint();

//    mrpt::math::CMatrixDouble44 h;
//    h <<   -1, 0, 0, 0,
//            0, -1, 0, -0,
//            0, 0, 1, -30,
//            0, 0, 0, 1;
//    mrpt::poses::CPose3D camPose = mrpt::poses::CPose3D(h);
//    mrpt::opengl::CCamera &cam = window.getDefaultViewport()->getCamera();
//    cam.set6DOFMode(true);
//    cam.setPose(camPose);

//    //2d plot
//    mrpt::gui::CDisplayWindowPlots  win("path2D", 600, 600);
//    win.hold_on();
//    win.setPos(1250, 0);
//    //mrpt::system::sleep(10000);

    for (unsigned num = 0; num < (str_left.size() - 1); num++)//str_left.size()
    {
        //get start timeb
        clock_t start_time = clock();
        if (initial == false)
        {
            //initial map
            GetImgandEqualHist(str_left[num], str_right[num], img_left, img_right);
            myslam::Frame::Ptr frame_ptr(new myslam::Frame(num, img_left, double(num), kitti,first_initial_left));
            map_ptr->InsertFrame(frame_ptr);
            map_ptr->InitialMap(img_left, img_right);
            double r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3;
            ifs >> r11 >> r12 >> r13 >> t1 >> r21 >> r22 >> r23 >> t2 >> r31 >> r32 >> r33 >> t3;
            initial = true;
        }
        else
        {
            GetImgandEqualHist(str_left[num], str_right[num], img_left2, img_right2);
            map_ptr->VOProcess(img_left, img_right, img_left2, img_right2);
            GetPoseLeftandRight(map_ptr->keyframes_[num]->T_c_w_ceres, tr, pose_temp);
            pose_left.push_back(pose_temp);
            cout<< "This is " << num << " pose:" <<endl;
            cout<<pose_temp.transpose()<<endl;
            fp1 << pose_temp[0]<< "  "<< pose_temp[1]<< "  "<< pose_temp[2]<< "  "
                               << pose_temp[3]<< "  "<< pose_temp[4]<< "  "<< pose_temp[5]<< endl;
/////////MRPT Visualization
//            std::vector<double> plot_x, plot_y, truth_x, truth_y;
//            plot_x.push_back(pose_left[num][0] * 0.001);
//            plot_y.push_back(pose_left[num][2] * 0.001);
//            win.plot(plot_x, plot_y, "r.3");
//            double r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3;
//            ifs >> r11 >> r12 >> r13 >> t1 >> r21 >> r22 >> r23 >> t2 >> r31 >> r32 >> r33 >> t3;
//            truth_x.push_back(t1);
//            truth_y.push_back(t3);
//            win.plot(truth_x, truth_y, "b.3");
//            win.axis_fit();
//            win.axis_equal(true);

//            cv::Mat rvec(3, 1, CV_64FC1), Rvec(3, 3, CV_64FC1), tvec(3, 1, CV_64FC1);
//            for(int i = 0; i < 3; i++)
//            {
//                tvec.at<double>(i) = map_ptr->keyframes_[num]->T_c_w_ceres[i];
//                rvec.at<double>(i) = map_ptr->keyframes_[num]->T_c_w_ceres[i + 3];
//            }
//            cv::Rodrigues(rvec, Rvec);
//            Rvec = Rvec.inv();
//            cv::Rodrigues(Rvec, rvec);
//            tvec = - Rvec * tvec;
//            mrpt::math::CMatrixDouble44 T_ck;
//            T_ck << Rvec.at<double>(0, 0), Rvec.at<double>(0, 1), Rvec.at<double>(0, 2), tvec.at<double>(0) * 0.001,
//                    Rvec.at<double>(1, 0), Rvec.at<double>(1, 1), Rvec.at<double>(1, 2), tvec.at<double>(1) * 0.001,
//                    Rvec.at<double>(2, 0), Rvec.at<double>(2, 1), Rvec.at<double>(2, 2), tvec.at<double>(2) * 0.001,
//                    0,       0,       0,       1;
////            T_ck << Rvec.at<double>(0, 0), 0                    , Rvec.at<double>(0, 2), tvec.at<double>(0) * 0.001,
////                    0                    , 1                    , 0                    , tvec.at<double>(1) * 0.001,
////                    Rvec.at<double>(2, 0), 0                    , Rvec.at<double>(2, 2), tvec.at<double>(2) * 0.001,
////                    0,                     0,                   0,                       1;
//            mrpt::poses::CPose3D cpose_ck = mrpt::poses::CPose3D (T_ck);
//            mrpt::maps::CColouredPointsMap points;

////            string str = str_disparity[num];
////            str.erase(str.end() - 1);
////            cv:: Mat disparity = cv::imread(str, 0);
////            for (int v = 0 ; v < disparity.rows; v+=8){
////                for (int u = 0; u < disparity.cols; u+=8){
////                    Eigen::Matrix<double, 4, 1> point4d;
////                    point4d[2] = 0.001 * tr * map_ptr->camera_->fx_ / ((double)(disparity.at<uchar>(v, u)));
////                    point4d[0] = point4d[2] * (u + 1 - map_ptr->camera_->cx_) / map_ptr->camera_->fx_;
////                    point4d[1] = point4d[2] * (v + 1 - map_ptr->camera_->cy_) / map_ptr->camera_->fy_;
////                    point4d[3] = 1;
////                    if (point4d[2] > 0 && point4d[1] < 2 && point4d[2] < 10){
////                        point4d = T_ck * point4d;
////                        //cout<<point4d.transpose()<<endl;
////                        points.insertPoint(point4d[0], point4d[1], point4d[2]);
////                    }
////                }
////            }

//            for (auto &mp : map_ptr->map_points_){
//                //if (mp.second->matched_frame_[0] == num){
//                    points.insertPoint(mp.second->pos_[0] * 0.001, mp.second->pos_[1] * 0.001, mp.second->pos_[2] * 0.001);
//                //}
//            }
//            //DenseMatch(img_left, img_right, map_ptr, points);
//            //pointsAll.addFrom(points);
//            IplImage *ts1 = cvLoadImage(str_left[num].c_str(), 0);
//            mrpt::utils::CImage zero;
//            zero.loadFromIplImage(ts1);
//            scene = window.get3DSceneAndLock();
//            vi->setImageView(zero);
//            kinectp->loadFromPointsMap<mrpt::maps::CColouredPointsMap> (&points);
//            window.unlockAccess3DScene();
//            window.repaint();

//            cam.setPose(cpose_ck + camPose);
//            viTop->getCamera().setPointingAt(cpose_ck.x(),cpose_ck.y(),cpose_ck.z());

//            scene = window.get3DSceneAndLock();
//            {
//                mrpt::opengl::CSetOfObjectsPtr reference = mrpt::opengl::stock_objects::CornerXYZ();
//                reference->setScale(0.5);
//                reference->setPose(cpose_ck);
//                scene->insert(reference);
//            }
//            window.unlockAccess3DScene();

        }
        clock_t  end_time = clock();
        double  sub_time = (end_time - start_time) * 0.000001;
        //cout << "cost time(s):  " << sub_time<< endl;
        time_sum += sub_time;
    }
    time_sum = time_sum / str_left.size();
    cout<<"mean time is:  "<< time_sum<< endl;

//    window.useCameraFromScene(false);
//    cam.set6DOFMode(false);
//    window.setCameraElevationDeg (0);
//    window.setCameraAzimuthDeg (-90);
//    window.setCameraPointingToPoint (150,0,150);
//    window.setCameraZoom (500);
//    window.setCameraProjective (true);
//    win.hold_off();

    ResultEstimation(pose_left, fp_truth);
    infile_left.close();
    infile_right.close();
    fp1.close();
    return 0;
}
inline void GetImgandEqualHist(string &str_left, string &str_right, cv::Mat& left, cv::Mat& right)
{
    str_left.erase(str_left.end() - 1); // delete /r
    str_right.erase(str_right.end() - 1); // delete /r
    left = cv::imread(str_left, 0);
    right = cv::imread(str_right, 0);
    thread thread1(cv::equalizeHist, left, left);
    thread thread2(cv::equalizeHist, right, right);
    thread1.join();
    thread2.join();
}

inline void GetPoseLeftandRight(const double T_c_w_ceres[6], const double tr, Vector6d &pose3)
{
    cv::Mat rvec(3, 1, CV_64FC1), Rvec(3, 3, CV_64FC1), tvec(3, 1, CV_64FC1);
    for(int i = 0; i < 3; i++)
    {
        tvec.at<double>(i) = T_c_w_ceres[i];
        rvec.at<double>(i) = T_c_w_ceres[i + 3];
    }
    cv::Rodrigues(rvec, Rvec);
    Rvec = Rvec.inv();
    cv::Rodrigues(Rvec, rvec);
    tvec = - Rvec * tvec;

    cv::Mat tr_vector = (cv::Mat_<double>(3, 1) << tr, 0, 0);
    cv::Mat tr_after = (cv::Mat_<double>(3, 1) << tr, 0, 0);
    tr_after = Rvec * tr_vector;

    pose3 << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2),
            rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2);
}

void MRPTShow(const cv::Mat &img,
              const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pose,
              const vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &truth,
              const myslam::Map::Ptr map)
{

}

void ResultEstimation(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pose_left, const string &fp_truth)
{
    ifstream ifs(fp_truth);
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> ground_truth, trajectory_estimate;
    for (int i = 0; i < pose_left.size(); i++)
    {
        double r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3;
        ifs >> r11 >> r12 >> r13 >> t1 >> r21 >> r22 >> r23 >> t2 >> r31 >> r32 >> r33 >> t3;
        Eigen::Matrix3d R;
        R << r11, r12, r13,
                r21, r22, r23,
                r31, r32, r33;
        Eigen::Vector3d t;
        t << t1, t2, t3;
        Sophus::SE3 T_truth(R, t);
        ground_truth.push_back(T_truth);
        double r_norm, r1, r2, r3;
        r_norm = sqrt(pose_left[i][0] * pose_left[i][0] + pose_left[i][1] * pose_left[i][1] + pose_left[i][2] * pose_left[i][2]);
        r1 = pose_left[i][0] / r_norm;
        r2 = pose_left[i][1] / r_norm;
        r3 = pose_left[i][2] / r_norm;
        Eigen::Matrix3d R_estimate = Eigen::AngleAxisd(r_norm, Eigen::Vector3d(r1, r2, r3)).toRotationMatrix();
        Eigen::Vector3d t_estimate;
        t_estimate << pose_left[i][0] * 0.001, pose_left[i][1] * 0.001, pose_left[i][2] * 0.001;
        Sophus::SE3 T_estimate(R_estimate, t_estimate);
        trajectory_estimate.push_back(T_estimate);
    }
    ifs.close();

    vector<double> error_t;
    double error_mean = 0, tra_length = 0,relative_error = 0;
    for (int i = 0; i < pose_left.size(); i++)
    {
        Eigen::Vector3d t = ground_truth[i].translation();
        error_t.push_back(sqrt(pow(t[0] - pose_left[i][0] * 0.001, 2) + pow(t[1] - pose_left[i][1] * 0.001, 2) + pow(t[2] - pose_left[i][2] * 0.001, 2)));
        error_mean += error_t[i];
    }
    error_mean = error_mean / pose_left.size();
    for (int i = 0; i < pose_left.size() - 1; i++)
    {
        Eigen::Vector3d t1 = ground_truth[i].translation();
        Eigen::Vector3d t2 = ground_truth[i + 1].translation();
        tra_length += sqrt(pow(t2[0] - t1[0], 2) + pow(t2[1] - t1[1], 2) + pow(t2[2] - t1[2], 2));
        relative_error += error_t[i + 1] / tra_length * 100;
    }
    relative_error = relative_error / (pose_left.size() - 1);

    cout<<"Average distance error is:  " << error_mean<<" m" << endl;
    cout<<"Average relative error is:  " << relative_error <<" %" << endl;
    DrawTrajectory(ground_truth, trajectory_estimate);
    cv::waitKey(0);
}

void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses2) {
    if (poses.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
                pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
                );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses.size() - 1; i++) {
            //glColor3f(1 - (float) i / poses.size(), 0.0f, (float) i / poses.size());
            glColor3f(0.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);

            glColor3f(1.0f, 0.0f, 1.0f);
            auto p3 = poses2[i], p4 = poses2[i + 1];
            glVertex3d(p3.translation()[0], p3.translation()[1], p3.translation()[2]);
            glVertex3d(p4.translation()[0], p4.translation()[1], p4.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

void DenseMatch(const cv::Mat &img_left,
                const cv::Mat &img_right,
                const myslam::Map::Ptr map1,
                mrpt::maps::CColouredPointsMap &point4d)
{
    vector<cv::KeyPoint> keypoints_1;
    cv::FAST(img_left, keypoints_1, 20, true);
    vector<cv::Point2f> pt1, pt2;
    for(int i = 0; i < keypoints_1.size(); i++)
        pt1.push_back(cv::Point2f(keypoints_1[i].pt.x, keypoints_1[i].pt.y));

    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img_left, img_right, pt1, pt2, status, error, cv::Size(5, 5), 4);
    Eigen::MatrixXd KeypointPose(pt1.size(), 3);
    myslam::Frame::Ptr frame_now = map1->keyframes_[map1->keyframes_.size() - 1];
    map1->Intersection(frame_now, frame_now, pt1, pt2, KeypointPose);
    for (int i = 0; i < pt1.size(); i++)
        point4d.insertPoint(KeypointPose(i, 0) * 0.001, KeypointPose(i, 1) * 0.001, KeypointPose(i, 2) * 0.001);
}
