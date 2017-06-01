#include "boost/date_time/posix_time/posix_time.hpp"
#include "opencv2/opencv.hpp"
#include "util/NumType.h"
#include "vio/Measurements.hpp"
#include "vio/Parameters.hpp"
#include "vio/kinematics/Transformation.hpp"
#include "vio/time/Time.hpp"
#include <Eigen/Dense>
using namespace boost::posix_time;
Eigen::Quaterniond RPY2Quat(const Eigen::Vector3d &rpy) {
  float roll = rpy(0), pitch = rpy(1), yaw = rpy(2);
  Eigen::Quaterniond q;
  q = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
  std::cout << "Quaternion" << std::endl << q.coeffs() << std::endl;
}

int main(int argc, char const *argv[]) {
  // ImuParameters imu_para;
  dsio::Time t = dsio::Time::now();
  double sec, nsec;
  sec = t.toSec();
  nsec = t.toNSec();
  std::cout << sec << '\t' << nsec << '\n';
  cv::Point pt(1, 2);

  dso::SE3 set = dso::SE3(Eigen::MatrixXd::Identity(4, 4));

  Eigen::Matrix3d m;
  m = Eigen::AngleAxisd(0.25 * M_PI, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(0.5 * M_PI, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(0.33 * M_PI, Eigen::Vector3d::UnitZ());
  std::cout << m << std::endl << "is unitary: " << m.isUnitary() << std::endl;

  Eigen::Quaterniond qd(m);
  set.setQuaternion(qd);
  set.translation() = Eigen::Vector3d(1, 10, 100);
  dsio::Transformation tr(set.matrix());
  std::cout << set.matrix() << '\n';
  std::cout << tr.C() << '\n' << tr.r() << '\n';
  std::cout << tr.T3x4() << '\n';
  Eigen::Vector3d vec31(1, 0, 0.1);
  std::cout << dsio::deltaQ(vec31).toRotationMatrix() << '\n';
  std::cout << RPY2Quat(Eigen::Vector3d(1, 0, 0.1)).toRotationMatrix() << '\n';
  std::cout << "right Jacobian" << '\n';
  std::cout << dsio::rightJacobian(vec31) << "\n\n";
  Sophus::SO3d rot3(dsio::deltaQ(vec31));
  std::cout << rot3.matrix() << '\n';
  return 0;
}
