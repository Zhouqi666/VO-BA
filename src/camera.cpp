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

#include "camera.h"
//#include <config.h>

namespace myslam
{
    
Camera::Camera()
{
    //fx_ = Config::get<float>("camera.fx");
    //fy_ = Config::get<float>("camera.fy");
    //cx_ = Config::get<float>("camera.cx");
    //cy_ = Config::get<float>("camera.cy");
}

Camera::Camera (const cv::Mat &camera_mat_, const cv::Mat &distCoeffs_, const double tr_, const int img_row, const int img_col)
{
    camera_mat = camera_mat_;
    distCoeffs = distCoeffs_;
    fx_ = camera_mat.at<double>(0, 0);
    fy_ = camera_mat.at<double>(1, 1);
    cx_ = camera_mat.at<double>(0, 2);
    cy_ = camera_mat.at<double>(1, 2);
    tr = tr_;
    img_row_ = img_row;
    img_col_ = img_col;
}


Eigen::Vector3d Camera::world2camera ( const Eigen::Vector3d& p_w, const double T_c_w[6] )
{
    cv::Mat r(3, 1, CV_64FC1), R(3, 3, CV_64FC1), t(3, 1, CV_64FC1), p_mat(3, 1, CV_64FC1);
    for(int i = 0; i < 3; i++)
	{
        p_mat.at<double>(i) = p_w(i);
        t.at<double>(i) = T_c_w[i];
        r.at<double>(i) = T_c_w[i + 3];
	}
    cv::Rodrigues(r, R);
    t += R * p_mat;

    Eigen::Vector3d p_camera;
    for(int i = 0; i < 3; i++)
        p_camera(i) = t.at<double>(i);
    return p_camera;
}

Eigen::Vector2d Camera::camera2pixel ( const Eigen::Vector3d& p_c )
{
    return Eigen::Vector2d (
               fx_ * p_c ( 0 ) / p_c ( 2 ) + cx_,
               fy_ * p_c ( 1 ) / p_c ( 2 ) + cy_
           );
}



Eigen::Vector2d Camera::world2pixel ( const Eigen::Vector3d& p_w, const double T_c_w[6] )
{
    return camera2pixel ( world2camera(p_w, T_c_w) );
}



}
