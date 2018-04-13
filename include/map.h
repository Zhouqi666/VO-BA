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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef MAP_H
#define MAP_H

#include "myslam.h"
#include "frame.h"
#include "mappoint.h"
#include "costfunction.hpp"
#include "resectcost.hpp"
#include "costfunction_first.hpp"

namespace myslam
{
class Map
{
public:
    typedef shared_ptr<Map> Ptr;
    unordered_map<unsigned long, myslam::MapPoint::Ptr>  map_points_;        // all landmarks
    unordered_map<unsigned long, myslam::MapPoint::Ptr>  mp_candidate_;        // all landmarks
    unordered_map<unsigned long, myslam::Frame::Ptr>     keyframes_;         // all key-frames
    Camera*                        camera_;     // Pinhole Camera model
    int ba_window_;
    double reproject_error_;

    Map();
    Map(Camera* camera, int ba_window , double reproject_error);

    void InsertFrame( myslam::Frame::Ptr frame);

    void InsertMapPoint( myslam::MapPoint::Ptr map_point );

    void InitialMap(const cv::Mat& img_left,
                    const cv::Mat& img_right
                    );

    void StereoUpdateMap(const cv::Mat& img_left,
                         const cv::Mat& img_right,
                         vector<cv::KeyPoint> &kp_left_far
                         );

    void StereoMatch(const cv::Mat &img_left,
                     const cv::Mat &img_right,
                     vector<cv::Point2f> &p_left_keypoint1,
                     vector<cv::Point2f> &p_right_keypoint1,
                     vector<cv::KeyPoint> &kp_left_far,
                     cv::Mat &descriptor_chosen
                     );

    void LBaseLineUpdate(vector<cv::KeyPoint> &kp_far
                     );

    void FarFrameMatch(const cv::Mat &img,
                       const cv::Mat &img_previous,
                       vector<cv::KeyPoint> &kp_far,
                       vector<cv::Point2f> &p_keypoint,
                       vector<cv::Point2f> &p_previous_keypoint,
                       cv::Mat &descriptor_chosen
                     );

    void Intersection(const myslam::Frame::Ptr frame1,
                      const myslam::Frame::Ptr frame2,
                      const vector<cv::Point2f> &p_left_keypoint,
                      const vector<cv::Point2f> &p_right_keypoint,
                      Eigen::MatrixXd &KeypointPose
                      );


    void MatchNextandUpdateMap(const cv::Mat &img_previous,
                               const cv::Mat &image_next
                               );

    void DirectPoseEstimationMultiLayer(
            const cv::Mat &img1,
            const cv::Mat &img2,
            const vector<cv::Point2f> &px_ref,
            const vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &p_world,
            Sophus::SE3 &T21
    );

    void DirectPoseEstimationSingleLayer(
            const cv::Mat &img1,
            const cv::Mat &img2,
            const vector<cv::Point2f> &px_ref,
            const vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &p_world,
            const double scale,
            Sophus::SE3 &T21
    );

    inline float GetPixelValue(const cv::Mat &img, float x, float y);

    void PoseOptimize();

    void BundleAdjustment();

    void EraseError();

    void EraseOutDate();

    void VOProcess(cv::Mat &img_left, cv::Mat &img_right, cv::Mat &img_left2, cv::Mat &img_right2);
};
}

#endif // MAP_H
//    //Direct VO
//    double t_initial[6] = {0, 0, 0, 0, 0, 0};
//    Sophus::SE3  T_delta;
//    auto last_frame = this->keyframes_.begin();
//    Eigen::Matrix3d R_previous = Eigen::AngleAxisd(0, Eigen::Vector3d(0, 0, 0)).toRotationMatrix();
//    double r_previous = sqrt(last_frame->second->T_c_w_ceres[3] * last_frame->second->T_c_w_ceres[3]
//                             + last_frame->second->T_c_w_ceres[4] * last_frame->second->T_c_w_ceres[4]
//                             + last_frame->second->T_c_w_ceres[5] * last_frame->second->T_c_w_ceres[5]);
//    if (r_previous != 0)
//    R_previous = Eigen::AngleAxisd(r_previous, Eigen::Vector3d(last_frame->second->T_c_w_ceres[3] / r_previous,
//                                                               last_frame->second->T_c_w_ceres[4] / r_previous,
//                                                               last_frame->second->T_c_w_ceres[5] / r_previous)).toRotationMatrix();
//    Sophus::SE3  T_Previous(R_previous, Eigen::Vector3d(last_frame->second->T_c_w_ceres[0], last_frame->second->T_c_w_ceres[1], last_frame->second->T_c_w_ceres[2]));

//    if (this->keyframes_.size() > 1)
//    {
//        double t_temp[6];
//        auto frame_last_two = this->keyframes_.begin();
//        frame_last_two++;
//        for (int i = 0; i < 6; i++)
//            t_temp[i] = last_frame->second->T_c_w_ceres[i] - frame_last_two->second->T_c_w_ceres[i];
//        double r_temp_norm = sqrt(t_temp[3] * t_temp[3] + t_temp[4] * t_temp[4] + t_temp[5] * t_temp[5]);
//        Eigen::Matrix3d R_temp = Eigen::AngleAxisd(r_temp_norm, Eigen::Vector3d(t_temp[3] / r_temp_norm, t_temp[4] / r_temp_norm, t_temp[5] / r_temp_norm)).toRotationMatrix();
//        Sophus::SE3 T_temp(R_temp, Eigen::Vector3d(t_temp[0], t_temp[1], t_temp[2]));
//        T_delta = T_temp;
//    }
//    DirectPoseEstimationMultiLayer(img_previous, img_next, pt1, pt1_cam, T_delta);
//    Sophus::SE3  T_next = T_delta * T_Previous;

//    Eigen::Matrix<double, 4, 4> Twc_matrix = T_next.matrix();
//    t_initial[0] = Twc_matrix(0, 3);
//    t_initial[1] = Twc_matrix(1, 3);
//    t_initial[2] = Twc_matrix(2, 3);
//    t_initial[3] = T_next.log()[3];
//    t_initial[4] = T_next.log()[4];
//    t_initial[5] = T_next.log()[5];
//    frame_next->setPose(t_initial);
//    this->InsertKeyFrame( frame_next );
//    double photometric_error = 0;
//    vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> KeypointPose;
//    for (int i = 0; i < pt1.size(); i++)
//    {
//        Eigen::Vector2d pixel_current = this->camera_->world2pixel(pt1_world[i], t_initial);
//        if (pixel_current[0] > 0 && pixel_current[0] < img_next.cols && pixel_current[1] > 0 && pixel_current[1] < img_next.rows)
//        {
//            double gray_previous = GetPixelValue(img_previous, pt1[i].x, pt1[i].y);
//            double gray_current = GetPixelValue(img_next, pixel_current[0], pixel_current[1]);
//            photometric_error = abs(gray_current - gray_previous);
//            if (photometric_error < 10)
//            {
//                Eigen::Matrix<double, 6, 1> temp;
//                temp[0] = pixel_current[0];
//                temp[1] = pixel_current[1];
//                temp[2] = this->map_points_[ mp_id[i] ]->pos_(0);
//                temp[3] = this->map_points_[ mp_id[i] ]->pos_(1);
//                temp[4] = this->map_points_[ mp_id[i] ]->pos_(2);
//                temp[5] = mp_id[i] ;
//                KeypointPose.push_back(temp);
//            }
//        }
//    }

//    for (unsigned i = 0; i < KeypointPose.size(); i++)
//    {
//        int mp_num = KeypointPose[i][5];
//        cv::Point2f pt_temp;
//        pt_temp.x = KeypointPose[i][0];
//        pt_temp.y = KeypointPose[i][1];
//        this->map_points_[mp_num]->matched_times_ += 1;
//        this->map_points_[mp_num]->matched_frame_.push_back(frame_next->id_);
//        this->map_points_[mp_num]->matched_pixel_.push_back(pt_temp);
//    }
