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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam.h"

namespace myslam
{

class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned long                id_;        // ID
    static unsigned long         factory_id_;    // factory id
    Eigen::Vector3d              pos_;       // Position in world
    double                       pos_ceres[3];

    int                          matched_times_;     // being an inliner in pose estimation
    vector<int>                  matched_frame_;
    vector<cv::Point2f>          matched_pixel_;

    cv::Mat                      descriptor_; // Descriptor for matching e
    bool                         good_;      // wheter a good point
    
    MapPoint();

    MapPoint( 
        unsigned long id, 
        const Eigen::Vector3d &position,
        const int             matched_frame,
        const cv::Point2f      &matched_pixel,
        const cv::Mat         &descriptor
    );
    
    static MapPoint::Ptr createMapPoint();

    static MapPoint::Ptr createMapPoint( 
        const Eigen::Vector3d &pos_world,
        const int             matched_frame,
        const cv::Point2f      &matched_pixel,
        const cv::Mat         &descriptor
         );

    void doubleCoverVector3d ();

    void Vector3dCoverdouble ();
};
}

#endif // MAPPOINT_H
