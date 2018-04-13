#include "myslam.h"
namespace myslam
{
struct FirstReprojectionError {
  FirstReprojectionError(double observed_x, double observed_y,double point_[3],double f)
      : observed_x(observed_x), observed_y(observed_y), f(f) {
        for(int i = 0; i < 3; i++)
          point[i] = point_[i];
  }

  template <typename T>
  bool operator()(const T* const camera,
                  T* residuals) const {
    // camera[3,4,5] are the angle-axis rotation.
    T p[3];
    T rotation[3];
    T point_t[3];

    for (int i = 0; i < 3; i++)
        point_t[i] = (T) point[i];

    rotation[0] = camera[3];
    rotation[1] = camera[4];
    rotation[2] = camera[5];

    ceres::AngleAxisRotatePoint(rotation, point_t, p);
    // camera[0,1,2] are the translation.
    p[0] += camera[0];
    p[1] += camera[1];
    p[2] += camera[2];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp =  p[0] / p[2];
    T yp =  p[1] / p[2];

    // Compute final projected point position.
    T const &focal = (T) f;
    T predicted_x = focal * xp;
    T predicted_y = focal * yp;
	//cout<<predicted_x<<"   "<<observed_x<<endl;
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);
    return true;
  }

  double observed_x;
  double observed_y;
  double point[3];
  double f;
};
}
