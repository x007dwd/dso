
#ifndef INCLUDE_OKVIS_MEASUREMENTS_HPP_
#define INCLUDE_OKVIS_MEASUREMENTS_HPP_

#include <deque>
#include <memory>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#include "vio/time/Time.hpp"
#include <Eigen/Dense>
/// \brief  Main namespace of this package.
namespace dsio {
/**
 * \brief Generic measurements
 *
 * They always come with a timestamp such that we can perform
 * any kind of asynchronous operation.
 * \tparam MEASUREMENT_T Measurement data type.
 */
template <class MEASUREMENT_T> struct Measurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Time timeStamp;            ///< Measurement timestamp
  MEASUREMENT_T measurement; ///< Actual measurement.
  int sensorId = -1; ///< Sensor ID. E.g. camera index in a multicamera setup

  /// \brief Default constructor.
  Measurement() : timeStamp(0.0) {}
  /**
   * @brief Constructor
   * @param timeStamp_ Measurement timestamp.
   * @param measurement_ Actual measurement.
   * @param sensorId Sensor ID (optional).
   */
  Measurement(const Time &timeStamp_, const MEASUREMENT_T &measurement_,
              int sensorId = -1)
      : timeStamp(timeStamp_), measurement(measurement_), sensorId(sensorId) {}
};

/// \brief IMU measurements. For now assume they are synchronized:
struct ImuSensorReadings {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief Default constructor.
  ImuSensorReadings() : gyroscopes(), accelerometers() {}
  /**
   * @brief Constructor.
   * @param gyroscopes_ Gyroscope measurement.
   * @param accelerometers_ Accelerometer measurement.
   */
  ImuSensorReadings(Eigen::Vector3d gyroscopes_,
                    Eigen::Vector3d accelerometers_)
      : gyroscopes(gyroscopes_), accelerometers(accelerometers_) {}
  Eigen::Vector3d gyroscopes;     ///< Gyroscope measurement.
  Eigen::Vector3d accelerometers; ///< Accelerometer measurement.
};

/// \brief Depth camera measurements. For now assume they are synchronized:
struct DepthCameraData {
  cv::Mat image;                 ///< Grayscale/RGB image.
  cv::Mat depthImage;            ///< Depth image.
  std::vector<cv::Point> points; ///< Keypoints if available.
  bool deliversKeypoints; ///< Are keypoints already delievered in measurement?
};

/// \brief Position measurement.
struct PositionReading {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d position;           ///< Position measurement.
  Eigen::Vector3d positionOffset;     ///< Position offset.
  Eigen::Matrix3d positionCovariance; ///< Measurement covariance.
};

/// \brief GPS position measurement.
struct GpsPositionReading {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double lat_wgs84;       ///< Latitude in WGS84 coordinate system.
  double lon_wgs84;       ///< Longitude in WGS84 coordiante system.
  double alt_wgs84;       ///< Altitude in WGS84 coordinate system.
  double geoidSeparation; ///< Separation between geoid (MSL) and WGS-84
                          /// ellipsoid. [m]
  Eigen::Matrix3d
      positionCovarianceENU; ///< Measurement covariance. East/North/Up.
};

/// \brief Magnetometer measurement.
struct MagnetometerReading {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ///< The magnetic flux density measurement. [uT]
  Eigen::Vector3d fluxDensity;
};

/// \brief Barometer measurement.
struct BarometerReading {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double pressure;    ///< Pressure measurement. [Pa]
  double temperature; ///< Temperature. [K]
};

/// \brief Differential pressure sensor measurement.
struct DifferentialPressureReading {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double pressure;                ///< Pressure measurement. [Pa]
  Eigen::Vector3d acceleration_B; ///< Acceleration in B-frame.
};

// this is how we store raw measurements before more advanced filling into
// data structures happens:
typedef Measurement<ImuSensorReadings> ImuMeasurement;
typedef std::deque<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement>>
    ImuMeasurementDeque;
/// \brief Camera measurement.
struct CameraData {
  cv::Mat image;                 ///< Image.
  std::vector<cv::Point> points; ///< Keypoints if available.
  bool deliversKeypoints;        ///< Are the keypoints delivered too?
};
/// \brief Keypoint measurement.
struct pointData {
  std::vector<cv::Point> points;              ///< Keypoints.
  std::vector<long unsigned int> landmarkIds; ///< Associated landmark IDs.
};
/// \brief Frame measurement.
struct FrameData {
  typedef std::shared_ptr<dsio::FrameData> Ptr;
  CameraData image;    ///< Camera measurement, i.e., image.
  pointData keypoints; ///< Keypoints.
};
typedef Measurement<CameraData> CameraMeasurement;
typedef Measurement<FrameData> FrameMeasurement;
typedef Measurement<DepthCameraData> DepthCameraMeasurement;

typedef Measurement<PositionReading> PositionMeasurement;
typedef std::deque<PositionMeasurement,
                   Eigen::aligned_allocator<PositionMeasurement>>
    PositionMeasurementDeque;

typedef Measurement<GpsPositionReading> GpsPositionMeasurement;
typedef std::deque<GpsPositionMeasurement,
                   Eigen::aligned_allocator<GpsPositionMeasurement>>
    GpsPositionMeasurementDeque;

typedef Measurement<MagnetometerReading> MagnetometerMeasurement;
typedef std::deque<MagnetometerMeasurement,
                   Eigen::aligned_allocator<MagnetometerMeasurement>>
    MagnetometerMeasurementDeque;

typedef Measurement<BarometerReading> BarometerMeasurement;

typedef Measurement<DifferentialPressureReading>
    DifferentialPressureMeasurement;

typedef Eigen::Matrix<double, 9, 1> SpeedAndBias;
}

#endif // INCLUDE_OKVIS_MEASUREMENTS_HPP_
