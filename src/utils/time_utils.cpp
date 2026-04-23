#include "pellet/utils/time_utils.hpp"

#include <chrono>

namespace pellet::utils {

int64_t NowMs() {
  using Clock = std::chrono::steady_clock;
  const auto now = Clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

}  // namespace pellet::utils
