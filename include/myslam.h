
//#pragma once

#ifndef MYSLAM_H
#define MYSLAM_H

//STD
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>
#include <map>
#include <math.h>
#include <thread>
#include <mutex>

//MRPT
#include <mrpt/system/threads.h>
#include <mrpt/system/os.h>
#include <mrpt/gui.h>
#include <mrpt/opengl.h>
#include <mrpt/maps/CColouredPointsMap.h>
#include <mrpt/math/CMatrixFixedNumeric.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/gui/CDisplayWindowPlots.h>
#include <mrpt/math/distributions.h>

//OPENCV
#include <opencv/cxcore.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/core/core.hpp>

//EIGEN
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

//CERES
#include <glog/logging.h>
#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <ceres/rotation.h>
#include <ceres/problem.h>

//SOPHUS
#include <sophus/se3.h>
#include <sophus/so3.h>

//PANGOLIN
#include <pangolin/pangolin.h>

using namespace std;
#endif
