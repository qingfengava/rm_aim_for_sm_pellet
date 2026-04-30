#pragma once

#include <chrono>
#include <cstdint>

#include "pellet/config.hpp"

namespace pellet::utils {

enum class DebugFeature : uint32_t {
  kCaptureLogs = 1u << 0,
  kPipelineStats = 1u << 1,
  kThreadStatus = 1u << 2,
  kStats1s = 1u << 3,
  kInferLogs = 1u << 4,
};

constexpr uint32_t DebugFeatureMask(DebugFeature feature) {
  return static_cast<uint32_t>(feature);
}

uint32_t ResolveDebugModules(const PelletConfig& config);
bool IsDebugEnabled(const PelletConfig& config, DebugFeature feature);
bool IsAnyDebugEnabled(const PelletConfig& config);
bool ShouldLogRateLimited(
    const char* module,
    const char* event_key,
    std::chrono::milliseconds interval = std::chrono::milliseconds(1000));

}  // namespace pellet::utils
