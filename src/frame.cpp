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

#include "frame.h"

namespace myslam
{
Frame::Frame()
    : id_(1), time_stamp_(-1), camera_(nullptr), is_key_frame_(true)
{

}

Frame::Frame (const long id, const cv::Mat &image, const double time_stamp, myslam::Camera* camera, double T_c_w[6])
    : id_(id), time_stamp_(time_stamp), camera_(camera), is_key_frame_(true)
{
    if (T_c_w != nullptr)
    {
        for(int i = 0; i < 6; i++)
        {
            T_c_w_ceres[i] = T_c_w[i];
            T_c_w_backup[i] = T_c_w[i];
        }
    }
    image_ = image.clone();
}

Frame::~Frame()
{
}

void Frame::setPose ( const double T_c_w[6] )
{
    for(int i = 0; i< 6; i++)
    {
        T_c_w_ceres[i] = T_c_w[i];
        T_c_w_backup[i] = T_c_w[i];
    }
}
void Frame::CancelBA()
{
    for (int i = 0; i < 6; i++)
        T_c_w_ceres[i] = T_c_w_backup[i];
}

void Frame::CoverBackup()
{
    for (int i = 0; i< 6; i++)
        T_c_w_backup[i] = T_c_w_ceres[i];
}

bool Frame::isinFrame ( const Eigen::Vector3d& pt_world )
{
    Eigen::Vector3d  p_cam = camera_->world2camera( pt_world, T_c_w_ceres );
    // cout<<"P_cam = "<<p_cam.transpose()<<endl;
    if ( p_cam(2) < 0 ) return false;
    Eigen::Vector2d pixel = camera_->world2pixel( pt_world, T_c_w_ceres );
    // cout<<"P_pixel = "<<pixel.transpose()<<endl<<endl;
    return pixel(0) > 0 && pixel(1) > 0
            && pixel(0) < this->camera_->img_row_
            && pixel(1) < this->camera_->img_col_;
}

void Frame::Tcw2Twc (Eigen::Matrix<double, 6, 1> &Twc_left)
{
    cv::Mat rvec(3, 1, CV_64FC1), Rvec(3, 3, CV_64FC1), tvec(3, 1, CV_64FC1);
    for(int i = 0; i < 3; i++)
    {
        tvec.at<double>(i) = this->T_c_w_ceres[i];
        rvec.at<double>(i) = this->T_c_w_ceres[i + 3];
    }
    cv::Rodrigues(rvec, Rvec);
    Rvec = Rvec.inv();
    cv::Rodrigues(Rvec, rvec);
    tvec = - Rvec * tvec;

    Twc_left[0] = tvec.at<double>(0);
    Twc_left[1] = tvec.at<double>(1);
    Twc_left[2] = tvec.at<double>(2);
    Twc_left[3] = rvec.at<double>(0);
    Twc_left[4] = rvec.at<double>(1);
    Twc_left[5] = rvec.at<double>(2);
}

void Frame::Tcw2TwcRight (Eigen::Matrix<double, 6, 1> &Twc_right)
{
    cv::Mat rvec(3, 1, CV_64FC1), Rvec(3, 3, CV_64FC1), tvec(3, 1, CV_64FC1);
    for(int i = 0; i < 3; i++)
    {
        tvec.at<double>(i) = this->T_c_w_ceres[i];
        rvec.at<double>(i) = this->T_c_w_ceres[i + 3];
    }
    cv::Rodrigues(rvec, Rvec);
    Rvec = Rvec.inv();
    cv::Rodrigues(Rvec, rvec);
    tvec = - Rvec * tvec;

    cv::Mat tr_vector = (cv::Mat_<double>(3, 1) << this->camera_->tr, 0, 0);
    cv::Mat tr_after = (cv::Mat_<double>(3, 1) << this->camera_->tr, 0, 0);
    tr_after = Rvec * tr_vector;
    Twc_right[0] = tvec.at<double>(0) + tr_after.at<double>(0);
    Twc_right[1] = tvec.at<double>(1) + tr_after.at<double>(1);
    Twc_right[2] = tvec.at<double>(2) + tr_after.at<double>(2);
    Twc_right[3] = rvec.at<double>(0);
    Twc_right[4] = rvec.at<double>(1);
    Twc_right[5] = rvec.at<double>(2);
}

}
