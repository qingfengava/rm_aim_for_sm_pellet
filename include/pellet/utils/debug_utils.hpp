#pragma once

#include <cstdint>

#include "pellet/config.hpp"

namespace pellet::utils {

enum class DebugFeature : uint32_t {
  kCaptureLogs = 1u << 0,
  kMorphology = 1u << 1,
  kPipelineStats = 1u << 2,
  kThreadStatus = 1u << 3,
  kShowWindow = 1u << 4,
  kInferLogs = 1u << 5,
};

constexpr uint32_t DebugFeatureMask(DebugFeature feature) {
  return static_cast<uint32_t>(feature);
}

uint32_t ResolveDebugModules(const PelletConfig& config);
bool IsDebugEnabled(const PelletConfig& config, DebugFeature feature);
bool IsAnyDebugEnabled(const PelletConfig& config);

}  // namespace pellet::utils
