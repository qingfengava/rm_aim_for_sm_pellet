#include "pellet/utils/debug_utils.hpp"

namespace pellet::utils {
namespace {

constexpr uint32_t kAllDebugMask =
    DebugFeatureMask(DebugFeature::kCaptureLogs) |
    DebugFeatureMask(DebugFeature::kMorphology) |
    DebugFeatureMask(DebugFeature::kPipelineStats) |
    DebugFeatureMask(DebugFeature::kThreadStatus) |
    DebugFeatureMask(DebugFeature::kShowWindow) |
    DebugFeatureMask(DebugFeature::kShowMask);

uint32_t LevelDebugMask(int level) {
  if (level <= 0) {
    return DebugFeatureMask(DebugFeature::kCaptureLogs);
  }
  if (level == 1) {
    return DebugFeatureMask(DebugFeature::kCaptureLogs) |
           DebugFeatureMask(DebugFeature::kPipelineStats) |
           DebugFeatureMask(DebugFeature::kThreadStatus);
  }
  return kAllDebugMask;
}

}  // namespace

uint32_t ResolveDebugModules(const PelletConfig& config) {
  if (config.debug.modules_mask != 0) {
    return config.debug.modules_mask;
  }
  if (config.debug.enable) {
    return LevelDebugMask(config.debug.level);
  }
  return 0;
}

bool IsDebugEnabled(const PelletConfig& config, DebugFeature feature) {
  return (ResolveDebugModules(config) & DebugFeatureMask(feature)) != 0;
}

bool IsAnyDebugEnabled(const PelletConfig& config) {
  return ResolveDebugModules(config) != 0;
}

}  // namespace pellet::utils
