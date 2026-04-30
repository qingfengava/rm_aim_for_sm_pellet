#include "pellet/utils/debug_utils.hpp"

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>

namespace pellet::utils {
namespace {

uint32_t SelectDebugModulesByMask(const uint32_t mask) {
  constexpr uint32_t kValidBits =
      DebugFeatureMask(DebugFeature::kCaptureLogs) |
      DebugFeatureMask(DebugFeature::kPipelineStats) |
      DebugFeatureMask(DebugFeature::kThreadStatus) |
      DebugFeatureMask(DebugFeature::kStats1s) |
      DebugFeatureMask(DebugFeature::kInferLogs);
  return mask & kValidBits;
}

using Clock = std::chrono::steady_clock;

std::mutex& RateLimitMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, Clock::time_point>& RateLimitTimestamps() {
  static std::unordered_map<std::string, Clock::time_point> timestamps;
  return timestamps;
}

}  // namespace

uint32_t ResolveDebugModules(const PelletConfig& config) {
  return SelectDebugModulesByMask(config.debug.modules_mask);
}

bool IsDebugEnabled(const PelletConfig& config, DebugFeature feature) {
  return (ResolveDebugModules(config) & DebugFeatureMask(feature)) != 0;
}

bool IsAnyDebugEnabled(const PelletConfig& config) {
  return ResolveDebugModules(config) != 0;
}

bool ShouldLogRateLimited(
    const char* module,
    const char* event_key,
    const std::chrono::milliseconds interval) {
  const auto effective_interval = std::max<std::chrono::milliseconds>(interval, std::chrono::milliseconds(0));
  if (effective_interval.count() == 0) {
    return true;
  }

  std::string key;
  if (module != nullptr && module[0] != '\0') {
    key += module;
  } else {
    key += "global";
  }
  key += "::";
  if (event_key != nullptr && event_key[0] != '\0') {
    key += event_key;
  } else {
    key += "default";
  }

  const auto now = Clock::now();
  std::lock_guard<std::mutex> lock(RateLimitMutex());
  auto& timestamps = RateLimitTimestamps();
  const auto it = timestamps.find(key);
  if (it != timestamps.end() && (now - it->second) < effective_interval) {
    return false;
  }
  timestamps[key] = now;
  return true;
}

}  // namespace pellet::utils
