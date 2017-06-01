
#ifndef INCLUDE_OKVIS_DURATION_HPP_
#define INCLUDE_OKVIS_DURATION_HPP_

/*********************************************************************
 ** Pragmas
 *********************************************************************/

#ifdef _MSC_VER
// Okvistime has some magic interface that doesn't directly include
// its implementation, this just disbales those warnings.
#pragma warning(disable : 4244)
#pragma warning(disable : 4661)
#endif

#include <climits>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <stdint.h>
//#include "rostime_decl.h"

/// \brief dsio Main namespace of this package.
namespace dsio {
void normalizeSecNSecSigned(int64_t &sec, int64_t &nsec);
void normalizeSecNSecSigned(int32_t &sec, int32_t &nsec);

/**
 * \brief Base class for Duration implementations.  Provides storage, common
 * functions and operator overloads.
 * This should not need to be used directly.
 */
template <class T> class DurationBase {
public:
  int32_t sec, nsec;
  DurationBase() : sec(0), nsec(0) {}
  DurationBase(int32_t _sec, int32_t _nsec);
  explicit DurationBase(double t) { fromSec(t); };
  ~DurationBase() {}
  T operator+(const T &rhs) const;
  T operator-(const T &rhs) const;
  T operator-() const;
  T operator*(double scale) const;
  T &operator+=(const T &rhs);
  T &operator-=(const T &rhs);
  T &operator*=(double scale);
  bool operator==(const T &rhs) const;
  inline bool operator!=(const T &rhs) const {
    return !(*static_cast<const T *>(this) == rhs);
  }
  bool operator>(const T &rhs) const;
  bool operator<(const T &rhs) const;
  bool operator>=(const T &rhs) const;
  bool operator<=(const T &rhs) const;
  double toSec() const { return (double)sec + 1e-9 * (double)nsec; };
  int64_t toNSec() const {
    return (int64_t)sec * 1000000000ll + (int64_t)nsec;
  };
  T &fromSec(double t);
  T &fromNSec(int64_t t);
  bool isZero();
};

class Rate;

/**
 * \brief Duration representation for use with the Time class.
 *
 * dsio::DurationBase provides most of its functionality.
 */
class Duration : public DurationBase<Duration> {
public:
  Duration() : DurationBase<Duration>() {}

  Duration(int32_t _sec, int32_t _nsec) : DurationBase<Duration>(_sec, _nsec) {}

  explicit Duration(double t) { fromSec(t); }
  explicit Duration(const Rate &);
  /**
   * \brief sleep for the amount of time specified by this Duration.  If a
   * signal interrupts the sleep, resleeps for the time remaining.
   */
  bool sleep() const;
};

extern const Duration DURATION_MAX;
extern const Duration DURATION_MIN;

/**
 * \brief Duration representation for use with the WallTime class.
 *
 * dsio::DurationBase provides most of its functionality.
 */
class WallDuration : public DurationBase<WallDuration> {
public:
  WallDuration() : DurationBase<WallDuration>() {}

  WallDuration(int32_t _sec, int32_t _nsec)
      : DurationBase<WallDuration>(_sec, _nsec) {}

  explicit WallDuration(double t) { fromSec(t); }
  explicit WallDuration(const Rate &);
  /**
   * \brief sleep for the amount of time specified by this Duration.  If a
   * signal interrupts the sleep, resleeps for the time remaining.
   */
  bool sleep() const;
};

std::ostream &operator<<(std::ostream &os, const Duration &rhs);
std::ostream &operator<<(std::ostream &os, const WallDuration &rhs);
}

#endif // INCLUDE_OKVIS_DURATION_HPP_
