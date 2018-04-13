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

#ifndef FRAME_H
#define FRAME_H

#include "myslam.h"
#include "camera.h"

namespace myslam 
{
    
// forward declare 
class MapPoint;
class Frame
{
public:
    typedef shared_ptr<Frame>      Ptr;
    unsigned long                  id_;         // id of this frame
    double                         time_stamp_; // when it is recorded
    cv::Mat                        image_;
    double                         T_c_w_ceres[6];
    double                         T_c_w_backup[6];
    Camera*                        camera_;
     //std::vector<cv::KeyPoint>      keypoints_;  // key points in image
     //std::vector<MapPoint*>         map_points_; // associated map points
    bool                           is_key_frame_;  // whether a key-frame
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
public: // data members 
    Frame();

    Frame( const long id,
           const cv::Mat &image,
           const double time_stamp = 0,
           Camera *camera_ = nullptr,
           double T_c_w[6] = nullptr
            );

    ~Frame();
    
    void setPose(const double T_c_w[6]);

    bool isinFrame(const Eigen::Vector3d& pos);

	void CancelBA();

	void CoverBackup();

    void Tcw2Twc (Eigen::Matrix<double, 6, 1> &Twc_left);

    void Tcw2TwcRight (Eigen::Matrix<double, 6, 1> &Twc_right);
    

};

}

#endif // FRAME_H
