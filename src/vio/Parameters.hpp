
#ifndef INCLUDE_OKVIS_PARAMETERS_HPP_
#define INCLUDE_OKVIS_PARAMETERS_HPP_
#include "util/NumType.h"
#include <deque>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop
#include "vio/kinematics/Transformation.hpp"
#include <Eigen/Dense>
namespace dsio {

/// \brief Struct to define the behavior of the camera extrinsics.
struct ExtrinsicsEstimationParameters {
  // set to 0 in order to turn off
  /// \brief Default Constructor -- fixed camera extrinsics.
  ExtrinsicsEstimationParameters()
      : sigma_absolute_translation(0.0), sigma_absolute_orientation(0.0),
        sigma_c_relative_translation(0.0), sigma_c_relative_orientation(0.0) {}

  /**
   * @brief Constructor.
   * @param sigma_absolute_translation Absolute translation stdev. [m]
   * @param sigma_absolute_orientation Absolute orientation stdev. [rad]
   * @param sigma_c_relative_translation Relative translation noise density.
   * [m/sqrt(Hz)]
   * @param sigma_c_relative_orientation Relative orientation noise density.
   * [rad/sqrt(Hz)]
   */
  ExtrinsicsEstimationParameters(double sigma_absolute_translation,
                                 double sigma_absolute_orientation,
                                 double sigma_c_relative_translation,
                                 double sigma_c_relative_orientation)
      : sigma_absolute_translation(sigma_absolute_translation),
        sigma_absolute_orientation(sigma_absolute_orientation),
        sigma_c_relative_translation(sigma_c_relative_translation),
        sigma_c_relative_orientation(sigma_c_relative_orientation) {}

  // absolute (prior) w.r.t frame S
  double sigma_absolute_translation; ///< Absolute translation stdev. [m]
  double sigma_absolute_orientation; ///< Absolute orientation stdev. [rad]

  // relative (temporal)
  double sigma_c_relative_translation; ///< Relative translation noise density.
                                       ///[m/sqrt(Hz)]
  double sigma_c_relative_orientation; ///< Relative orientation noise density.
                                       ///[rad/sqrt(Hz)]
};

/*!
 * \brief IMU parameters.
 *
 * A simple struct to specify properties of an IMU.
 *
 */
struct ImuParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Transformation
      T_BS;         ///< Transformation from Body frame to IMU (sensor frame S).
  double a_max;     ///< Accelerometer saturation. [m/s^2]
  double g_max;     ///< Gyroscope saturation. [rad/s]
  double sigma_g_c; ///< Gyroscope noise density.
  double sigma_bg;  ///< Initial gyroscope bias.
  double sigma_a_c; ///< Accelerometer noise density.
  double sigma_ba;  ///< Initial accelerometer bias
  double sigma_gw_c;  ///< Gyroscope drift noise density.
  double sigma_aw_c;  ///< Accelerometer drift noise density.
  double tau;         ///< Reversion time constant of accerometer bias. [s]
  double g;           ///< Earth acceleration.
  Eigen::Vector3d a0; ///< Mean of the prior accelerometer bias.
  int rate;           ///< IMU rate in Hz.
};

/*!
 * \brief Magnetometer parameters.
 *
 * A simple struct to specify properties of a magnetometer.
 *
 */
struct MagnetometerParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double stdev; ///< Measurement (white noise part) standard deviation. [uT]
  double priorStdev;      ///< Prior. [uT]
  double tau;             ///< Reversion time constant of bias [s]
  double sigma_c;         ///< Bias noise density [uT/sqrt(Hz)]
  double updateFrequency; ///< Related state estimates are inserted at this
                          /// frequency. [Hz]
};

/*!
 * \brief GPS parameters
 *
 * A simple struct to specify properties of a GPS receiver.
 *
 */
struct GpsParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d antennaOffset; ///< The position offset of the antenna in body
                                 ///(B) coordinates.
};

/*!
 * \brief Position sensor parameters.
 *
 * A simple struct to specify properties of a position sensor.
 *
 */
struct PositionSensorParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d positionSensorOffset; ///< The position offset of the position
                                        /// sensor in body (B) coordinates.
  bool isLeveled; ///< If true, the position sensor measurements are assumed to
                  /// be world z up (exactly, i.e. only yaw gets estimated).
};

/*!
 * \brief Magnetic ENU z bias
 *
 * A simple struct to specify the dynamics of magnetic ENU z component
 * variation.
 *
 */
struct MagneticEnuZParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double priorStdev;      ///< ENU z-component bias prior. [uT]
  double sigma_c;         ///< ENU z-component noise density. [uT/s/sqrt(Hz)]
  double tau;             ///< Reversion time constant of ENU z-component. [s]
  double updateFrequency; ///< Related state estimates are inserted at this
                          /// frequency. [Hz]
};

/*!
 * \brief Barometer parameters.
 *
 * A simple struct to specify properties of a barometer.
 *
 */
struct BarometerParameters {
  double staticPressureStdev; ///< Measurement (white noise part) standard
                              /// deviation. [kPa]
  double temperatureStdev;    ///< Measurement (white noise part) standard
                              /// deviation. [kPa]
};

/*!
 * \brief QFF parameters.
 *
 * A simple struct to specify the QFF state dynamics.
 *
 */
struct QffParameters {
  double priorStdev;      ///< Prior of QFF [kPa]
  double sigma_c;         ///< Drift noise density. [kPa/sqrt(s)]
  double tau;             ///< Reversion time constant. [s]
  double updateFrequency; ///< Related state estimates are inserted at this
                          /// frequency. [Hz]
};

/*!
 * \brief Differential pressure sensor parameters.
 *
 * A simple struct to specify properties of a differential pressure sensor.
 *
 */
struct DifferentialPressureSensorParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double differentialPressureStdev; ///< Measurement (white noise part) standard
                                    /// deviation. [Pa]
  /// Measurement (white noise part) standard deviation [m/s^2]
  /// Be conservative, this has to account for model uncertainties, too...
  double accelerationStdev;
  /// Polynomial for -z direction (dimensionless) force, alpha deg.
  /// polynomial vector of lenght n as (Matlab polyval style):
  /// c[0]*x^(n-1)+c[1]*x^(n-2)+...+c[n-2]*x+c[n-1] .
  Eigen::VectorXd c_minus_z_of_alpha;
  double c_y_of_beta; ///< Polynomial (only proportionality) for y direction
                      ///(dimensionless) force, beta deg.
  double m;           ///< Mass. [kg]
  double A;           ///< Reference wing area. [m^2]
};

/*!
 * \brief Wind parameters.
 *
 * A simple struct to specify the wind state dynamics.
 *
 */
struct WindParameters {
  double priorStdev;      ///< Prior of wind. [m/s]
  double sigma_c;         ///< Drift noise density. [m/s/sqrt(s)]
  double tau;             ///< Reversion time constant. [s];
  double updateFrequency; ///< Related state estimates are inserted at this
                          /// frequency. [Hz]
};

/**
 * @brief Parameters for optimization and related things (detection).
 */
struct Optimization {
  int max_iterations; ///< Maximum iterations the optimization should perform.
  int min_iterations; ///< Minimum iterations the optimization should perform.
  double timeLimitForMatchingAndOptimization; ///< The time limit for both
                                              /// matching and optimization. [s]
  double timeReserve;        ///< Store a little more on the beginning and end
                             /// of the IMU buffer. [s]
  double detectionThreshold; ///< Keypoint detection threshold.
  bool useMedianFilter;      ///< Use a Median filter over captured image?
  int detectionOctaves;      ///< Number of keypoint detection octaves.
  int maxNoKeypoints; ///< Restrict to a maximum of this many keypoints per
                      /// image (strongest ones).
  int numKeyframes;   ///< Number of keyframes.
  int numImuFrames;   ///< Number of IMU frames.
};

/**
 * @brief Information on camera and IMU setup.
 */
struct SensorsInformation {
  int cameraRate;    ///< Camera rate in Hz.
  double imageDelay; ///< Camera image delay. [s]
  int imuIdx; ///< IMU index. Anything other than 0 will probably not work.
  double frameTimestampTolerance; ///< Time tolerance between frames to accept
                                  /// them as stereo frames. [s]
};

/// @brief Some visualization settings.
struct Visualization {
  bool displayImages; ///< Display images?
};

enum class FrameName { B, S, W, Wc };

/// @brief Some publishing parameters.
struct PublishingParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int publishRate = 200; ///< Maximum publishing rate. [Hz]
  bool publishLandmarks =
      true; ///< Select, if you want to publish landmarks at all.
  float landmarkQualityThreshold = 1.0e-5; ///< Quality threshold under which
                                           /// landmarks are not published.
  /// Between 0 and 1.
  float maxLandmarkQuality = 0.05; ///< Quality above which landmarks are
  /// assumed to be of the best quality. Between
  /// 0 and 1.
  size_t maxPathLength =
      100; ///< Maximum length of ros::nav_mgsgs::Path to be published.
  bool publishImuPropagatedState =
      true; ///< Should the state that is propagated with IMU messages be
            /// published? Or just the optimized ones?
  Transformation T_Wc_W = Transformation::Identity(); ///< Provide custom World
                                                      /// frame Wc
  FrameName trackedBodyFrame = FrameName::B; ///< B or S, the frame of reference
                                             /// that will be expressed relative
  /// to the selected worldFrame Wc
  FrameName velocitiesFrame =
      FrameName::B; ///< B or S,  the frames in which the velocities of the
                    /// selected trackedBodyFrame will be expressed in
};

/// @brief Struct to combine all parameters and settings.
struct VioParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Optimization optimization;   ///< Optimization parameters.
  Visualization visualization; ///< Visualization parameters.
  SensorsInformation
      sensors_information; ///< Information on camera and IMU setup.
  ExtrinsicsEstimationParameters
      camera_extrinsics; ///< Camera extrinsic estimation parameters.
  ImuParameters imu;     ///< IMU parameters
  MagnetometerParameters magnetometer; ///< Magnetometer parameters.
  PositionSensorParameters position;   ///< Position sensor parameters.
  GpsParameters gps;                   ///< GPS parameters
  MagneticEnuZParameters
      magnetic_enu_z; ///< Dynamics of magnetic ENU z component variation.
  BarometerParameters barometer; ///< Barometer parameters.
  QffParameters qff;             ///< QFF parameters.
  DifferentialPressureSensorParameters
      differential;                ///< Differential pressure sensor parameters.
  WindParameters wind;             ///< Wind parameters.
  PublishingParameters publishing; ///< Publishing parameters.
};

} // namespace okvis

#endif // INCLUDE_OKVIS_PARAMETERS_HPP_
