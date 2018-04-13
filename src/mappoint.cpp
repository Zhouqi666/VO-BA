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

#include "myslam.h"
#include "mappoint.h"

namespace myslam
{

MapPoint::MapPoint()
    : id_(-1), pos_(Eigen::Vector3d(0,0,0)), matched_frame_(-1), good_(true), matched_times_(0)
{

}

MapPoint::MapPoint ( long unsigned int id, const Eigen::Vector3d& position, const int matched_frame, const cv::Point2f &matched_pixel, const cv::Mat &descriptor)
: id_(id), pos_(position), good_(true), descriptor_(descriptor), matched_times_(1)
{
    matched_frame_.push_back(matched_frame);
	matched_pixel_.push_back(matched_pixel);
    for (int i = 0; i < 3; i++)
        pos_ceres[i] = position(i);
}

MapPoint::Ptr MapPoint::createMapPoint()
{
    return MapPoint::Ptr( 
        new MapPoint()
    ); 
}

MapPoint::Ptr MapPoint::createMapPoint ( 
        const Eigen::Vector3d  &pos_world,
        const int              matched_frame,
        const cv::Point2f      &matched_pixel,
        const cv::Mat          &descriptor)
{
    return MapPoint::Ptr( 
        new MapPoint( factory_id_++, pos_world, matched_frame, matched_pixel, descriptor )
    );
}

void MapPoint::doubleCoverVector3d()
{
    for (int i = 0; i < 3; i++)
    pos_(i) = pos_ceres[i];

}

void MapPoint::Vector3dCoverdouble()
{
    for (int i = 0; i < 3; i++)
    pos_ceres[i] = pos_(i);

}


unsigned long MapPoint::factory_id_ = 0;

}
