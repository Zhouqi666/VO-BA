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

#ifndef CAMERA_H
#define CAMERA_H

#include "myslam.h"

namespace myslam
{

// Pinhole camera model
class Camera
{
public:
    //typedef shared_ptr<Camera> Ptr;
    float   fx_, fy_, cx_, cy_;
    int img_row_;
    int img_col_;
    double tr;//baseline
    cv::Mat camera_mat;
    cv::Mat distCoeffs;


    Camera();

    Camera (const cv::Mat &camera_mat_,
            const cv::Mat &distCoeffs_,
            const double tr,
            const int img_row,
            const int img_col
            );

    Eigen::Vector3d world2camera( const Eigen::Vector3d& p_w,
                                  const double T_c_w[6]
                                  );

    Eigen::Vector2d camera2pixel ( const Eigen::Vector3d& p_c );

    Eigen::Vector2d world2pixel ( const Eigen::Vector3d& p_w, const double T_c_w[6] );

};

}
#endif // CAMERA_H
