/*
* <one line to give the program's name and a brief idea of what it does.>
* Copyright (C) 2016  <copyright holder> <email>
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*/

#include "map.h"

namespace myslam
{
Map::Map() {
    map_points_.rehash(1000);
    keyframes_.rehash(4550);
    ba_window_ = 5;
}

Map::Map(Camera* camera, int ba_window, double reproject_error){
    map_points_.rehash(1000);
    keyframes_.rehash(4550);
    ba_window_ = ba_window;
    camera_ = camera;
    reproject_error_ = reproject_error;
}


void Map::InsertFrame ( Frame::Ptr frame)
{
    if ( keyframes_.find(frame->id_) == keyframes_.end() )
    {
        keyframes_.insert( make_pair(frame->id_, frame) );
    }
    else
    {
        keyframes_[ frame->id_ ] = frame;
    }
}

void Map::InsertMapPoint ( MapPoint::Ptr map_point )
{
    if ( map_points_.find(map_point->id_) == map_points_.end() )
    {
        map_points_.insert( make_pair(map_point->id_, map_point) );
    }
    else
    {
        map_points_[map_point->id_] = map_point;
    }
}

void Map::VOProcess(cv::Mat &img_left, cv::Mat &img_right, cv::Mat &img_left2, cv::Mat &img_right2)
{
    //LK track the next frame
    this->MatchNextandUpdateMap(img_left, img_left2);
    //pose optimize
    if (this->keyframes_.begin()->second->is_key_frame_ == true)
    {
        this->PoseOptimize();

        //bundle adjustment
        this->BundleAdjustment();

        //update map
        vector<cv::KeyPoint> kp_far;
        this->StereoUpdateMap(img_left2, img_right2, kp_far);
        if (kp_far.size() > 5)
            this->LBaseLineUpdate(kp_far);

        //erase out_of_date mappoints
        this->EraseOutDate();
    }

    img_left = img_left2.clone();//i+1
    img_right = img_right2.clone();
}

void Map::InitialMap(const cv::Mat& img_left,
                     const cv::Mat& img_right
                     )
{
    vector<cv::Point2f> p_left_keypoint, p_right_keypoint ;
    vector<cv::KeyPoint> useless_kpoints;
    cv::Mat descriptors_1_chosen;

    this->StereoMatch(img_left, img_right, p_left_keypoint, p_right_keypoint, useless_kpoints, descriptors_1_chosen);
    Eigen::MatrixXd KeypointPose(p_left_keypoint.size(), 3);
    myslam::Frame::Ptr frame_now = this->keyframes_[this->keyframes_.size() - 1];
    this->Intersection(frame_now, frame_now, p_left_keypoint, p_right_keypoint, KeypointPose);

    for (unsigned i = 0; i < p_left_keypoint.size(); i++)
    {
        Eigen::Vector3d mapp_pose(0, 0, 0);
        mapp_pose(0) = KeypointPose(i, 0);
        mapp_pose(1) = KeypointPose(i, 1);
        mapp_pose(2) = KeypointPose(i, 2);
        MapPoint::Ptr mp = MapPoint::createMapPoint(mapp_pose, frame_now->id_, p_left_keypoint[i], descriptors_1_chosen.row(i));
        this->InsertMapPoint(mp);
    }

}

void Map::MatchNextandUpdateMap(const cv::Mat &img_previous,
                                const cv::Mat &img_next)
{
    //Project mappoints
    this->EraseError();
    vector<cv::Point2f> pt1, pt2;
    vector<int> mp_id;
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pt1_world, pt1_cam;
    cout<< "The number of map points is:  " << this->map_points_.size() << endl;
    int last_frame_num = this->keyframes_.begin()->second->id_;

    myslam::Frame::Ptr frame_next(new Frame(last_frame_num + 1, img_next, double(last_frame_num + 1),this->camera_));

    for (auto &mp : this->map_points_)
    {
        Eigen::Vector3d pos_world = mp.second->pos_;

        double T_c_w[6];
        for (int i = 0; i < 6; ++i)
            T_c_w[i] = this->keyframes_[last_frame_num]->T_c_w_ceres[i]; //latest frame

        Eigen::Vector2d pos_pixel = this->camera_->world2pixel(pos_world, T_c_w);
        Eigen::Vector3d pos_cam = this->camera_->world2camera(pos_world, T_c_w);
        cv::Point2f pt_temp;
        pt_temp.x = pos_pixel[0];
        pt_temp.y = pos_pixel[1];
        if (pt_temp.x > 20 && pt_temp.x < img_previous.cols - 20 && pt_temp.y > 20 && pt_temp.y < img_previous.rows - 20 )
        {
            pt1.push_back(pt_temp);
            mp_id.push_back(mp.second->id_);
            pt1_world.push_back(pos_world);
            pt1_cam.push_back(pos_cam);
        }
    }

    //    //LK according to velocity mm
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img_previous, img_next, pt1, pt2, status, error, cv::Size(7, 7), 5);

    //RANSAC
    int ptCount = (int)pt1.size();
    int success_count = 0;
    cv::Mat p1(ptCount, 2, CV_32F);
    cv::Mat p2(ptCount, 2, CV_32F);
    for (int i = 0; i < ptCount; i++)
    {
        p1.at<float>(i, 0) = pt1[i].x;
        p1.at<float>(i, 1) = pt1[i].y;
        p2.at<float>(i, 0) = pt2[i].x;
        p2.at<float>(i, 1) = pt2[i].y;
    }
    cv::Mat m_Fundamental;
    vector<uchar> m_RANSACStatus;
    m_Fundamental = cv::findFundamentalMat(p1, p2, m_RANSACStatus, cv::FM_RANSAC, 1.0, 0.99);
    int trueNum = 0;
    for (unsigned i = 0; i < m_RANSACStatus.size(); i++)
        if (m_RANSACStatus[i] == true)
            trueNum ++;

    Eigen::MatrixXd KeypointPose(trueNum, 6);
    for( unsigned i = 0; i < pt1.size(); i++ )
    {
        if (m_RANSACStatus[i] == true)
        {
            KeypointPose(success_count, 0) = pt2[i].x;
            KeypointPose(success_count, 1) = pt2[i].y;
            KeypointPose(success_count, 2) = this->map_points_[ mp_id[i] ]->pos_(0);
            KeypointPose(success_count, 3) = this->map_points_[ mp_id[i] ]->pos_(1);
            KeypointPose(success_count, 4) = this->map_points_[ mp_id[i] ]->pos_(2);
            KeypointPose(success_count++, 5) = mp_id[i] ;
        }
    }

    vector<cv::Point3f> list_points3d_model_match(KeypointPose.rows());
    vector<cv::Point2f> list_points2d_scene_match(KeypointPose.rows());
    for (int i = 0; i < KeypointPose.rows(); i++)
    {
        list_points2d_scene_match[i].x = KeypointPose(i, 0);
        list_points2d_scene_match[i].y = KeypointPose(i, 1);
        list_points3d_model_match[i].x = KeypointPose(i, 2);
        list_points3d_model_match[i].y = KeypointPose(i, 3);
        list_points3d_model_match[i].z = KeypointPose(i, 4);
    }

    vector<int> pnp_inlier;
    cv::Mat rvec(3, 1, CV_64FC1), tvec(3, 1, CV_64FC1);
    cv::solvePnPRansac(list_points3d_model_match,
                       list_points2d_scene_match,
                       frame_next->camera_->camera_mat,
                       frame_next->camera_->distCoeffs,
                       rvec,
                       tvec,
                       false,
                       100,
                       3.0f,
                       trueNum * 0.5,
                       pnp_inlier,
                       CV_EPNP);
    //frame
    double nth_initialpose[6];
    for (int i = 0; i < 3; i++)
    {
        nth_initialpose[i] = tvec.at<double>(i);
        nth_initialpose[i + 3] = rvec.at<double>(i);
    }
    auto keyframe_last = this->keyframes_.begin();
    while (keyframe_last->second->is_key_frame_ == false)
        keyframe_last++;

    //judge keyframe
    double t_delta = sqrt(pow(keyframe_last->second->T_c_w_ceres[0] - nth_initialpose[0], 2) +
            pow(keyframe_last->second->T_c_w_ceres[1] - nth_initialpose[1], 2) +
            pow(keyframe_last->second->T_c_w_ceres[2] - nth_initialpose[2], 2));

    double r_delta = sqrt(pow(keyframe_last->second->T_c_w_ceres[3] - nth_initialpose[3], 2) +
            pow(keyframe_last->second->T_c_w_ceres[4] - nth_initialpose[4], 2) +
            pow(keyframe_last->second->T_c_w_ceres[5] - nth_initialpose[5], 2));
    if (t_delta < 100 && r_delta < 0.01)
        frame_next->is_key_frame_ == false;
    frame_next->setPose( nth_initialpose);
    this->InsertFrame( frame_next );

    //put inliers into the map
    if (frame_next->is_key_frame_ == true)
    {
        for (unsigned i = 0; i < pnp_inlier.size(); i++)
        {
            int mp_num = KeypointPose(pnp_inlier[i], 5);
            cv::Point2f pt_temp;
            pt_temp.x = KeypointPose(pnp_inlier[i], 0);
            pt_temp.y = KeypointPose(pnp_inlier[i], 1);
            this->map_points_[mp_num]->matched_times_ += 1;
            this->map_points_[mp_num]->matched_frame_.push_back(frame_next->id_);
            this->map_points_[mp_num]->matched_pixel_.push_back(pt_temp);
        }
    }
}

void Map::DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const vector<cv::Point2f> &px_ref,
        const vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &p_world,
        Sophus::SE3 &T21
        ) {

    // parameters
    int pyramids = 4;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        cv::Mat temp1, temp2;
        cv::resize(img1, temp1, cv::Size(0, 0), scales[i], scales[i], cv::INTER_LINEAR);
        cv::resize(img2, temp2, cv::Size(0, 0), scales[i], scales[i], cv::INTER_LINEAR);
        pyr1.push_back(temp1);
        pyr2.push_back(temp2);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        vector<cv::Point2f> px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, p_world, scales[level], T21);
    }
}

inline float Map::GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
                (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
            );
}

void Map::DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const vector<cv::Point2f> &px_ref,
        const vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &p_world,
        const double scale,
        Sophus::SE3 &T21
        ) {

    // parameters
    double cx = scale * this->camera_->cx_;
    double cy = scale * this->camera_->cy_;
    double fx = scale * this->camera_->fx_;
    double fy = scale * this->camera_->fy_;
    int half_patch_size = 4;
    int iterations = 30;

    double cost = 0, lastCost = 0;
    int nGood = 0;  // good projections
    vector<cv::Point2f> goodProjection;

    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0;
        goodProjection.clear();

        // Define Hessian and bias
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();  // 6x6 Hessian
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();  // 6x1 bias
        for (size_t i = 0; i < px_ref.size(); i++) {

            // compute the projection in the second image
            float u = 0, v = 0;
            Sophus::Vector4d P, P2;
            P[0] = p_world[i][0];
            P[1] = p_world[i][1];
            P[2] = p_world[i][2];
            P[3] = 1;
            P2 = T21.matrix() * P;
            u = P2[0] * fx / P2[2] + cx;
            v = P2[1] * fy / P2[2] + cy;
            if (u > half_patch_size && u < img2.cols - half_patch_size && v > half_patch_size && v < img2.rows - half_patch_size){
                nGood++;
                goodProjection.push_back(cv::Point2f(u, v));
                // and compute error and jacobian
                for (int x = -half_patch_size; x < half_patch_size; x++)
                    for (int y = -half_patch_size; y < half_patch_size; y++) {

                        double error = 0;
                        error = GetPixelValue(img1, px_ref[i].x + x, px_ref[i].y + y)
                                - GetPixelValue(img2, u + x, v + y);
                        Sophus::Vector3d Pxy;
                        Pxy[0] = p_world[i][2] * (px_ref[i].x + x - cx) / fx;
                        Pxy[1] = p_world[i][2] * (px_ref[i].y + y - cy) / fy;
                        Pxy[2] = p_world[i][2];

                        Eigen::Matrix<double, 2, 6> J_pixel_xi;   // pixel to \xi in Lie algebra
                        Eigen::Vector2d J_img_pixel;    // image gradients
                        J_pixel_xi(0,0) = fx / Pxy[2];
                        J_pixel_xi(0,1) = 0;
                        J_pixel_xi(0,2) = - fx * Pxy[0] / (Pxy[2] * Pxy[2]);
                        J_pixel_xi(0,3) = -fx * Pxy[0] * Pxy(1) / (Pxy[2] * Pxy[2]);
                        J_pixel_xi(0,4) = fx + fx * Pxy[0] * Pxy[0] / (Pxy[2] * Pxy[2]);
                        J_pixel_xi(0,5) = - fx * Pxy(1) / Pxy[2];
                        J_pixel_xi(1,0) = 0;
                        J_pixel_xi(1,1) = fy / Pxy[2];
                        J_pixel_xi(1,2) = - fy * Pxy(1) / (Pxy[2] * Pxy[2]);
                        J_pixel_xi(1,3) = -fy - fy * Pxy(1) * Pxy(1)/(Pxy[2] * Pxy[2]);
                        J_pixel_xi(1,4) = fy * Pxy[0] * Pxy(1)/(Pxy[2] * Pxy[2]);
                        J_pixel_xi(1,5) = fy * Pxy[0] / Pxy[2];

                        J_img_pixel[0] = 0.5 * (GetPixelValue(img2, u + x + 1, v + y) - GetPixelValue(img2, u + x - 1, v + y));
                        J_img_pixel[1] = 0.5 * (GetPixelValue(img2, u + x, v + y + 1) - GetPixelValue(img2, u + x, v + y - 1));
                        // total jacobian
                        Eigen::Matrix<double, 6, 1> J;
                        J = - J_img_pixel.transpose() * J_pixel_xi;

                        H += J * J.transpose();
                        b += -error * J;
                        cost += error * error;
                    }
            }
        }
        // solve update and put it into estimation
        Eigen::Matrix<double, 6, 1> update;
        update = H.ldlt().solve(b);
        T21 = Sophus::SE3::exp(update) * T21;

        cost /= nGood;
        if (iter > 0 && cost > lastCost) {
            break;
        }
        lastCost = cost;
    }

}

void Map::StereoUpdateMap(const cv::Mat& img_left,
                          const cv::Mat& img_right,
                          vector<cv::KeyPoint> &kp_left_far
                          )
{
    vector<cv::Point2f> p_left_keypoint, p_right_keypoint;
    cv::Mat descriptors_1_chosen;

    this->StereoMatch(img_left, img_right,p_left_keypoint, p_right_keypoint, kp_left_far, descriptors_1_chosen);
    Eigen::MatrixXd KeypointPose(p_left_keypoint.size(), 3);
    myslam::Frame::Ptr frame_now = this->keyframes_[this->keyframes_.size() - 1];
    this->Intersection(frame_now, frame_now, p_left_keypoint, p_right_keypoint, KeypointPose);

    for (unsigned i = 0; i < p_left_keypoint.size(); i++)
    {
        Eigen::Vector3d mapp_pose(0, 0, 0);
        mapp_pose(0) = KeypointPose(i, 0);
        mapp_pose(1) = KeypointPose(i, 1);
        mapp_pose(2) = KeypointPose(i, 2);
        MapPoint::Ptr mp = MapPoint::createMapPoint(mapp_pose, frame_now->id_, p_left_keypoint[i], descriptors_1_chosen.row(i));
        this->InsertMapPoint(mp);
    }
}

void Map::StereoMatch(const cv::Mat &img_left,
                      const cv::Mat &img_right,
                      vector<cv::Point2f> &p_left_keypoint1,
                      vector<cv::Point2f> &p_right_keypoint1,
                      vector<cv::KeyPoint> &kp_left_far,
                      cv::Mat &descriptor_chosen)
{
    //cv::initModule_nonfree();
    int row = 3, col = 12;
    cv::Size mat_size;
    mat_size.width = img_left.cols / col;
    mat_size.height = img_left.rows / row;
    vector<cv::KeyPoint> keypoints_1;
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            int x = j * mat_size.width;
            int y = i * mat_size.height;
            cv::Mat img_temp = img_left(cv::Range(y, y + mat_size.height), cv::Range(x, x + mat_size.width));
            vector<cv::KeyPoint> keypoints_temp;
            double threshold = 120;
            cv::FAST(img_temp, keypoints_temp, threshold, true);
            while (keypoints_temp.size() < 15 && threshold > 40)
            {
                keypoints_temp.clear();
                threshold -= 20;
                cv::FAST(img_temp, keypoints_temp, threshold, true);
            }
            for (int k = 0; k < keypoints_temp.size(); k++){
                keypoints_temp[k].pt.x += x;
                keypoints_temp[k].pt.y += y;
                keypoints_1.push_back(keypoints_temp[k]);
            }
        }
    }
    vector<cv::Point2f> pt1, pt2;
    cv::Mat descriptors_1;
    cv::OrbDescriptorExtractor orb;
    orb.compute(img_left, keypoints_1, descriptors_1);
    for(int i = 0; i < keypoints_1.size(); i++)
        pt1.push_back(cv::Point2f(keypoints_1[i].pt.x, keypoints_1[i].pt.y));

    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img_left, img_right, pt1, pt2, status, error, cv::Size(6, 6), 5);

    double far_threshold = this->keyframes_.begin()->second->camera_->fx_ / 90;//100 times baseline
    double close_threshold = this->keyframes_.begin()->second->camera_->fx_ / 15;

    for (unsigned i = 0; i < pt1.size(); i++)
    {
        cv::Point2f col_delta = pt1[i] - pt2[i];
        if (col_delta.x > far_threshold && col_delta.x < close_threshold && abs(col_delta.y) < 1.0)
        {
            p_left_keypoint1.push_back(pt1[i]);
            p_right_keypoint1.push_back(pt2[i]);
            descriptor_chosen.push_back(descriptors_1.row(i));
        }
        else if (col_delta.x < far_threshold && col_delta.x > 1 && abs(col_delta.y) < 1.0)
        {
            kp_left_far.push_back(keypoints_1[i]);
        }
    }
}

void Map::Intersection(const myslam::Frame::Ptr frame1,
                       const myslam::Frame::Ptr frame2,
                       const vector<cv::Point2f> &p_left_keypoint,
                       const vector<cv::Point2f> &p_right_keypoint,
                       Eigen::MatrixXd &KeypointPose)
{

    double f = frame1->camera_->fx_;
    double cx1 = frame1->camera_->cx_;
    double cy1 = frame1->camera_->cy_;
    double cx2 = frame2->camera_->cx_;
    double cy2 = frame2->camera_->cy_;

    Eigen::Matrix<double, 6, 1> Twc_1, Twc_2;
    if(frame1 == frame2)
    {
        frame1->Tcw2Twc(Twc_1);
        frame2->Tcw2TwcRight(Twc_2);
    }
    else
    {
        frame1->Tcw2Twc(Twc_1);
        frame2->Tcw2Twc(Twc_2);
    }
    double Xs1 = Twc_1[0];
    double Xs2 = Twc_2[0];
    double Ys1 = Twc_1[1];
    double Ys2 = Twc_2[1];
    double Zs1 = Twc_1[2];
    double Zs2 = Twc_2[2];

    cv::Mat rvec1(3, 1, CV_64FC1), rvec2(3, 1, CV_64FC1);
    cv::Mat d1(3, 3, CV_64FC1), d2(3, 3, CV_64FC1);
    rvec1.at<double>(0) = Twc_1[3];
    rvec1.at<double>(1) = Twc_1[4];
    rvec1.at<double>(2) = Twc_1[5];
    cv::Rodrigues(rvec1, d1);
    rvec2.at<double>(0) = Twc_2[3];
    rvec2.at<double>(1) = Twc_2[4];
    rvec2.at<double>(2) = Twc_2[5];
    cv::Rodrigues(rvec2, d2);

    double a1 = d1.at<double>(0, 0);
    double a2 = d1.at<double>(0, 1);
    double a3 = d1.at<double>(0, 2);
    double b1 = d1.at<double>(1, 0);
    double b2 = d1.at<double>(1, 1);
    double b3 = d1.at<double>(1, 2);
    double c1 = d1.at<double>(2, 0);
    double c2 = d1.at<double>(2, 1);
    double c3 = d1.at<double>(2, 2);

    double a4 = d2.at<double>(0, 0);
    double a5 = d2.at<double>(0, 1);
    double a6 = d2.at<double>(0, 2);
    double b4 = d2.at<double>(1, 0);
    double b5 = d2.at<double>(1, 1);
    double b6 = d2.at<double>(1, 2);
    double c4 = d2.at<double>(2, 0);
    double c5 = d2.at<double>(2, 1);
    double c6 = d2.at<double>(2, 2);

    for (unsigned int i = 0; i < p_left_keypoint.size(); i++)
    {
        Eigen::MatrixXd L0(4, 3);
        Eigen::VectorXd L1(4);
        Eigen::MatrixXd X;
        L0(0, 0) = - f * a1 + a3 * (p_left_keypoint[i].x - cx1);
        L0(0, 1) = - f * b1 + b3 * (p_left_keypoint[i].x - cx1);
        L0(0, 2) = - f * c1 + c3 * (p_left_keypoint[i].x - cx1);
        L0(1, 0) = - f * a2 + a3 * (p_left_keypoint[i].y - cy1);
        L0(1, 1) = - f * b2 + b3 * (p_left_keypoint[i].y - cy1);
        L0(1, 2) = - f * c2 + c3 * (p_left_keypoint[i].y - cy1);

        L0(2, 0) = - f * a4 + a6 * (p_right_keypoint[i].x - cx2);
        L0(2, 1) = - f * b4 + b6 * (p_right_keypoint[i].x - cx2);
        L0(2, 2) = - f * c4 + c6 * (p_right_keypoint[i].x - cx2);
        L0(3, 0) = - f * a5 + a6 * (p_right_keypoint[i].y - cy2);
        L0(3, 1) = - f * b5 + b6 * (p_right_keypoint[i].y - cy2);
        L0(3, 2) = - f * c5 + c6 * (p_right_keypoint[i].y - cy2);

        L1(0) = - (f * a1 * Xs1 + f * b1 * Ys1 + f * c1 * Zs1) + a3 * Xs1 * (p_left_keypoint[i].x - cx1) + b3 * Ys1 * (p_left_keypoint[i].x - cx1) + c3 * Zs1 * (p_left_keypoint[i].x - cx1);
        L1(1) = - (f * a2 * Xs1 + f * b2 * Ys1 + f * c2 * Zs1) + a3 * Xs1 * (p_left_keypoint[i].y - cy1) + b3 * Ys1 * (p_left_keypoint[i].y - cy1) + c3 * Zs1 * (p_left_keypoint[i].y - cy1);

        L1(2) = - (f * a4 * Xs2 + f * b4 * Ys2 + f * c4 * Zs2) + a6 * Xs2 * (p_right_keypoint[i].x - cx2) + b6 * Ys2 * (p_right_keypoint[i].x - cx2) + c6 * Zs2 * (p_right_keypoint[i].x - cx2);
        L1(3) = - (f * a5 * Xs2 + f * b5 * Ys2 + f * c5 * Zs2) + a6 * Xs2 * (p_right_keypoint[i].y - cy2) + b6 * Ys2 * (p_right_keypoint[i].y - cy2) + c6 * Zs2 * (p_right_keypoint[i].y - cy2);
        X = L0.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(L1);
        KeypointPose(i, 0) = X(0);
        KeypointPose(i, 1) = X(1);
        KeypointPose(i, 2) = X(2);
    }
}

void Map::LBaseLineUpdate(vector<cv::KeyPoint> &kp_far)
{
    myslam::Frame::Ptr frame_now, frame_previous;
    frame_now = this->keyframes_[this->keyframes_.size() - 1];
    double T_now[6], T_previous[6];
    for (int i = 0; i < 6; i ++)
        T_now[i] = frame_now->T_c_w_ceres[i];
    double baseline = 0, delta_r = 0;
    auto iter = this->keyframes_.begin();
    iter++;
    int num = 0;
    while (baseline < 1000 && delta_r < 0.1 && iter != this->keyframes_.end())
    {
        if(iter->second->is_key_frame_ == true)
        {
            for (int i = 0; i < 6; i ++)
                T_previous[i]= iter->second->T_c_w_ceres[i];
            baseline = sqrt(pow(T_previous[0] - T_now[0], 2) + pow(T_previous[1] - T_now[1], 2) + pow(T_previous[2] - T_now[2], 2));
            delta_r = sqrt(pow(T_previous[3] - T_now[3], 2) + pow(T_previous[4] - T_now[4], 2) + pow(T_previous[5] - T_now[5], 2));
            frame_previous = iter->second;
            num++;
        }
        iter++;
    }
    //    if(delta_r < 0.1)
    //    {
    vector<cv::Point2f> p_keypoint, p_previous_keypoint;
    cv::Mat descriptors_1_chosen;
    this->FarFrameMatch(frame_now->image_, frame_previous->image_, kp_far, p_keypoint, p_previous_keypoint, descriptors_1_chosen);
    Eigen::MatrixXd KeypointPose(p_keypoint.size(), 3);
    this->Intersection(frame_now, frame_previous, p_keypoint, p_previous_keypoint, KeypointPose);

    for (unsigned i = 0; i < p_keypoint.size(); i++)
    {
        Eigen::Vector3d mapp_pose(0, 0, 0);
        mapp_pose(0) = KeypointPose(i, 0);
        mapp_pose(1) = KeypointPose(i, 1);
        mapp_pose(2) = KeypointPose(i, 2);
        MapPoint::Ptr mp = MapPoint::createMapPoint(mapp_pose, frame_now->id_, p_keypoint[i], descriptors_1_chosen.row(i));
        this->InsertMapPoint(mp);
    }
    //    }


}

void Map::FarFrameMatch(const cv::Mat &img,
                        const cv::Mat &img_previous,
                        vector<cv::KeyPoint> &kp_far,
                        vector<cv::Point2f> &p_keypoint,
                        vector<cv::Point2f> &p_previous_keypoint,
                        cv::Mat &descriptor_chosen)
{
    //cv::initModule_nonfree();
    cv::Mat descriptors_1;
    cv::OrbDescriptorExtractor orb;
    orb.compute(img, kp_far, descriptors_1);

    vector<uchar> status;
    vector<float> error;
    vector<cv::Point2f> pt1, pt2;
    for (int i = 0; i < kp_far.size(); i++)
        pt1.push_back(kp_far[i].pt);
    cv::calcOpticalFlowPyrLK(img, img_previous, pt1, pt2, status, error, cv::Size(7, 7), 5);

    cv::Mat m_Fundamental;
    vector<uchar> m_RANSACStatus;
    m_Fundamental = cv::findFundamentalMat(pt1, pt2, m_RANSACStatus, cv::FM_RANSAC, 1, 0.99);

    for (unsigned i = 0; i < pt1.size(); i++)
    {
        if (m_RANSACStatus[i] == 1)
        {
            p_keypoint.push_back(pt1[i]);
            p_previous_keypoint.push_back(pt2[i]);
            descriptor_chosen.push_back(descriptors_1.row(i));
        }
    }

}

void Map::PoseOptimize()
{
    this->EraseError();
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1);
    double f = this->camera_->fx_;
    int frame_now_id = this->keyframes_[this->keyframes_.size() - 1]->id_;
    for (auto &iter : this->map_points_)
    {
        int last_frame = iter.second->matched_times_ - 1;
        if(iter.second->matched_frame_[last_frame] == frame_now_id)
        {
            double cx = iter.second->matched_pixel_[last_frame].x - this->keyframes_[iter.second->matched_frame_[last_frame]]->camera_->cx_;
            double cy = iter.second->matched_pixel_[last_frame].y - this->keyframes_[iter.second->matched_frame_[last_frame]]->camera_->cy_;
            myslam::FirstReprojectionError *fResidual = new myslam::FirstReprojectionError(cx, cy, iter.second->pos_ceres, f);
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction< myslam::FirstReprojectionError, 2, 6 >(fResidual), loss_function, this->keyframes_[iter.second->matched_frame_[last_frame]]->T_c_w_ceres);
        }
    }
    ceres::Solver::Options m_options;
    ceres::Solver::Summary m_summary;
    m_options.max_num_iterations = 10;
    m_options.linear_solver_type = ceres::DENSE_SCHUR;//DENSE_SCHUR
    m_options.num_threads = 1;
    //m_options.max_solver_time_in_seconds = 0.005;
    //m_options.minimizer_progress_to_stdout = true;

    ceres::Solve(m_options, &problem, &m_summary);

    //fprintf(stdout,"%s\n",m_summary.BriefReport().c_str());

    auto iterLast = this->keyframes_[frame_now_id];
    iterLast->CoverBackup();

}

void Map::BundleAdjustment()
{
    clock_t start_time = clock();
    //this->EraseError();
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1);
    double f = this->camera_->fx_;
    auto last_keyframe = this->keyframes_.begin();
    int keyframe_num = 1, keyframe_last_id = 0;
    while (keyframe_num < ba_window_ && last_keyframe != this->keyframes_.end())
    {
        if (last_keyframe->second->is_key_frame_ == true)
        {
            keyframe_num++;
            keyframe_last_id = last_keyframe->second->id_;
        }
        last_keyframe++;
    }

    for (auto &iter : this->map_points_)
    {
        if(iter.second->matched_frame_[0] <= keyframe_last_id)
        {
            for(int i = 1; i < iter.second->matched_times_; i++)
            {
                double cx = iter.second->matched_pixel_[i].x - this->keyframes_[iter.second->matched_frame_[i]]->camera_->cx_;
                double cy = iter.second->matched_pixel_[i].y - this->keyframes_[iter.second->matched_frame_[i]]->camera_->cy_;
                myslam::FirstReprojectionError *fResidual = new myslam::FirstReprojectionError(cx, cy, iter.second->pos_ceres, f);
                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<myslam::FirstReprojectionError, 2, 6>(fResidual), loss_function, this->keyframes_[iter.second->matched_frame_[i]]->T_c_w_ceres);
            }
        }
        else //if (iter.second->matched_frame_[0] != frame_now_id)
        {
            for(int i = 0; i < iter.second->matched_times_; i++)
            {
                double cx = iter.second->matched_pixel_[i].x - this->keyframes_[iter.second->matched_frame_[i]]->camera_->cx_;
                double cy = iter.second->matched_pixel_[i].y - this->keyframes_[iter.second->matched_frame_[i]]->camera_->cy_;
                myslam::ReprojectionError *Residual = new myslam::ReprojectionError(cx, cy, f);
                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<myslam::ReprojectionError, 2, 6, 3>(Residual), loss_function, this->keyframes_[iter.second->matched_frame_[i]]->T_c_w_ceres,iter.second->pos_ceres);
            }
        }
    }
    ceres::Solver::Options m_options;
    ceres::Solver::Summary m_summary;
    m_options.max_num_iterations = 7;
    m_options.linear_solver_type = ceres::SPARSE_SCHUR;//DENSE_SCHUR
    //int num_threads = thread::hardware_concurrency();
    m_options.num_threads = 1;
    //m_options.max_solver_time_in_seconds = 0.025;

    ceres::Solve(m_options, &problem, &m_summary);

    //fprintf(stdout,"%s\n",m_summary.BriefReport().c_str());

    for (auto &iter : this->map_points_)
        iter.second->doubleCoverVector3d();
    for (auto &iter : this->keyframes_)
        iter.second->CoverBackup();
    clock_t  end_time = clock();
    double  sub_time = (end_time - start_time) * 0.000001;
    cout << " bundle adjustment cost time(s):  " << sub_time<< endl;

}

void Map::EraseError()
{
    vector<int> erase_points;
    //cout<<"total  "<< this->map_points_.size() <<" map points"<<endl;
    for (auto &iter : this->map_points_)
    {
        for(int i = 0; i < iter.second->matched_times_; i++)
        {
            int frame_num = iter.second->matched_frame_[i];
            cv::Point2f point_image = iter.second->matched_pixel_[i];
            double frame_pos[6];
            for(int j = 0; j < 6; j++)
                frame_pos[j] = this->keyframes_[frame_num]->T_c_w_ceres[j];
            double p[3], point[3], rotation[3];
            for(int k = 0; k < 3; k++)
                point[k] = iter.second->pos_ceres[k];

            rotation[0] = frame_pos[3];
            rotation[1] = frame_pos[4];
            rotation[2] = frame_pos[5];
            ceres::AngleAxisRotatePoint(rotation, point, p);
            p[0] += frame_pos[0];
            p[1] += frame_pos[1];
            p[2] += frame_pos[2];
            double xp =  p[0] / p[2];
            double yp =  p[1] / p[2];
            double focal = this->keyframes_[frame_num]->camera_->fx_;
            double predicted_x = focal * xp + this->keyframes_[frame_num]->camera_->cx_;
            double predicted_y = focal * yp + this->keyframes_[frame_num]->camera_->cy_;
            double residuals[2];
            residuals[0] = predicted_x - point_image.x;
            residuals[1] = predicted_y - point_image.y;

            double distance = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            double residual = sqrt(residuals[0] * residuals[0] + residuals[1] * residuals[1]);
            if (residual * distance > reproject_error_ * 10000)
            {
                erase_points.push_back(iter.second->id_);
                break;
            }
        }
    }
    for (unsigned i = 0; i < erase_points.size(); i++)
        this->map_points_.erase(erase_points[i]);
    //cout<<"total  "<< this->map_points_.size() <<" map points"<<endl;
}

void Map::EraseOutDate()
{
    auto last_keyframe = this->keyframes_.begin();
    int keyframe_num = 1, keyframe_last_id = 0;
    while (keyframe_num < ba_window_ && last_keyframe != this->keyframes_.end())
    {
        if (last_keyframe->second->is_key_frame_ == true)
        {
            keyframe_num++;
            keyframe_last_id = last_keyframe->second->id_;
        }
        last_keyframe++;
    }
    vector<int> erase_points;
    for (auto &iter : this->map_points_)
    {
        //cout<<iter.second->matched_frame_[0]<<endl;
        if (iter.second->matched_frame_[0] < keyframe_last_id)
            erase_points.push_back(iter.second->id_);
    }
    for (unsigned i = 0; i < erase_points.size(); i++)
        this->map_points_.erase(erase_points[i]);
}

}
//    clock_t start_time = clock();
//    double threshold = 80;
//    if (this->map_points_.size() < 150 * ba_window_)
//        threshold = 30;
//    else if (this->map_points_.size() < 300 * ba_window_)
//        threshold = 50;
//    vector<cv::KeyPoint> keypoints_1, keypoints_2;
//    cv::Mat descriptors_1, descriptors_2;
//    cv::OrbDescriptorExtractor orb;
//    thread t1(cv::FAST, img_left, ref(keypoints_1), threshold, true);
//    thread t2(cv::FAST, img_right, ref(keypoints_2), threshold, true);
//    t1.join();
//    t2.join();
//    thread t3(&cv::OrbDescriptorExtractor::compute, orb, ref(img_left), ref(keypoints_1), ref(descriptors_1));
//    thread t4(&cv::OrbDescriptorExtractor::compute, orb, ref(img_right), ref(keypoints_2), ref(descriptors_2));
//    t3.join();
//    t4.join();
//    cv::BFMatcher matcher(cv::NORM_HAMMING2);//NORM_HAMMING2
//    vector<cv::DMatch> matches, good_matches;
//    matcher.match(descriptors_1, descriptors_2, matches);

//    //RANSAC
//    int ptCount = (int)matches.size();
//    cv::Mat p1(ptCount, 2, CV_32F);
//    cv::Mat p2(ptCount, 2, CV_32F);
//    double mean = 0;
//    for (int i = 0; i < ptCount; i++)
//    {
//        p1.at<float>(i, 0) = keypoints_1[matches[i].queryIdx].pt.x;
//        p1.at<float>(i, 1) = keypoints_1[matches[i].queryIdx].pt.y;
//        p2.at<float>(i, 0) = keypoints_2[matches[i].trainIdx].pt.x;
//        p2.at<float>(i, 1) = keypoints_2[matches[i].trainIdx].pt.y;
//        mean += matches[i].distance;
//    }
//    mean = mean / ptCount;

//    cv::Mat m_Fundamental;
//    vector<uchar> m_RANSACStatus;
//    m_Fundamental = cv::findFundamentalMat(p1, p2, m_RANSACStatus, cv::FM_RANSAC, 1, 0.99);
//    double far_threshold = this->keyframes_.begin()->second->camera_->fx_ / 70;//60 times baseline

//    for (unsigned i = 0; i < matches.size(); i++)
//    {
//        cv::Point2f col_delta = keypoints_1[matches[i].queryIdx].pt - keypoints_2[matches[i].trainIdx].pt;
//        if (m_RANSACStatus[i] == 1 && matches[i].distance < 50 && col_delta.x > far_threshold && abs(col_delta.y) < 3)
//        {
//            good_matches.push_back(matches[i]);
//        }
//        else if (col_delta.x < far_threshold && col_delta.x > 0)
//        {
//            kp_left_far.push_back(keypoints_1[matches[i].queryIdx]);
//        }
//    }

//    for (unsigned i = 0; i < good_matches.size(); i++)
//    {
//        p_left_keypoint1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
//        p_right_keypoint1.push_back(keypoints_2[good_matches[i].trainIdx].pt);
//        descriptor_chosen.push_back(descriptors_1.row(good_matches[i].queryIdx));
//    }
//    clock_t  end_time = clock();
//    double  sub_time = (end_time - start_time) * 0.000001;
//    cout << " stereo match cost time(s):  " << sub_time<< endl;
