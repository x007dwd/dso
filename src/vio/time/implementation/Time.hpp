
#ifndef INCLUDE_OKVIS_IMPLEMENTATION_TIME_HPP_
#define INCLUDE_OKVIS_IMPLEMENTATION_TIME_HPP_

/*********************************************************************
 ** Headers
 *********************************************************************/

//#include <ros/platform.h>
#include <cmath>
#include <iostream>
//#include <ros/exception.h>

/*********************************************************************
 ** Cross Platform Headers
 *********************************************************************/

#ifdef WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

/// \brief dsio Main namespace of this package.
namespace dsio {

template <class T, class D> T &TimeBase<T, D>::fromNSec(uint64_t t) {
  sec = (int32_t)(t / 1000000000);
  nsec = (int32_t)(t % 1000000000);

  normalizeSecNSec(sec, nsec);

  return *static_cast<T *>(this);
}

template <class T, class D> D TimeBase<T, D>::operator-(const T &rhs) const {
  return D((int32_t)sec - (int32_t)rhs.sec,
           (int32_t)nsec - (int32_t)rhs.nsec); // carry handled in ctor
}

template <class T, class D> T TimeBase<T, D>::operator-(const D &rhs) const {
  return *static_cast<const T *>(this) + (-rhs);
}

template <class T, class D> T TimeBase<T, D>::operator+(const D &rhs) const {
  int64_t sec_sum = (int64_t)sec + (int64_t)rhs.sec;
  int64_t nsec_sum = (int64_t)nsec + (int64_t)rhs.nsec;

  // Throws an exception if we go out of 32-bit range
  normalizeSecNSecUnsigned(sec_sum, nsec_sum);

  // now, it's safe to downcast back to uint32 bits
  return T((uint32_t)sec_sum, (uint32_t)nsec_sum);
}

template <class T, class D> T &TimeBase<T, D>::operator+=(const D &rhs) {
  *this = *this + rhs;
  return *static_cast<T *>(this);
}

template <class T, class D> T &TimeBase<T, D>::operator-=(const D &rhs) {
  *this += (-rhs);
  return *static_cast<T *>(this);
}

template <class T, class D>
bool TimeBase<T, D>::operator==(const T &rhs) const {
  return sec == rhs.sec && nsec == rhs.nsec;
}

template <class T, class D> bool TimeBase<T, D>::operator<(const T &rhs) const {
  if (sec < rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec < rhs.nsec)
    return true;
  return false;
}

template <class T, class D> bool TimeBase<T, D>::operator>(const T &rhs) const {
  if (sec > rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec > rhs.nsec)
    return true;
  return false;
}

template <class T, class D>
bool TimeBase<T, D>::operator<=(const T &rhs) const {
  if (sec < rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec <= rhs.nsec)
    return true;
  return false;
}

template <class T, class D>
bool TimeBase<T, D>::operator>=(const T &rhs) const {
  if (sec > rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec >= rhs.nsec)
    return true;
  return false;
}
}

#endif // INCLUDE_OKVIS_IMPLEMENTATION_TIME_HPP_
