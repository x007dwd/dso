#include "boost/date_time/posix_time/posix_time.hpp"

namespace dso {
class Time {
public:
  Time(uint32_t _sec, uint32_t _nsec);

  /**
   * \brief Retrieve the current time.  Returns the current wall clock time.
   */
  static Time now();
  /**
   * \brief Sleep until a specific time has been reached.
   */
  static bool sleepUntil(const Time &end);

  static void init();
  static void shutdown();
  static void setNow(const Time &new_now);
  static bool useSystemTime();
  static bool isSimTime();
  static bool isSystemTime();

  /**
   * \brief Returns whether or not the current time is valid.  Time is valid if
   * it is non-zero.
   */
  static bool isValid();
  /**
   * \brief Wait for time to become valid
   */
  static bool waitForValid();
  /**
   * \brief Wait for time to become valid, with timeout
   */
  static bool waitForValid();

private:
  ptime ptm;
};
}
