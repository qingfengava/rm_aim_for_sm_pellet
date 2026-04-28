#include <gtest/gtest.h>

#include "pellet/config.hpp"
#include "pellet/utils/debug_utils.hpp"

namespace {

using pellet::utils::DebugFeature;
using pellet::utils::DebugFeatureMask;

TEST(DebugUtilsTest, DisabledDebugReturnsZeroMask) {
  pellet::PelletConfig config;
  config.debug.enable = false;
  config.debug.level = 2;
  config.debug.modules_mask = 0U;

  EXPECT_EQ(pellet::utils::ResolveDebugModules(config), 0U);
  EXPECT_FALSE(pellet::utils::IsAnyDebugEnabled(config));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kCaptureLogs));
}

TEST(DebugUtilsTest, LevelOneEnablesCapturePipelineAndThreadStatus) {
  pellet::PelletConfig config;
  config.debug.enable = true;
  config.debug.level = 1;
  config.debug.modules_mask = 0U;

  EXPECT_TRUE(pellet::utils::IsAnyDebugEnabled(config));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kCaptureLogs));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kPipelineStats));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kThreadStatus));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kMorphology));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kShowWindow));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kInferLogs));
}

TEST(DebugUtilsTest, ModulesMaskHasHigherPriorityThanLevel) {
  pellet::PelletConfig config;
  config.debug.enable = true;
  config.debug.level = 2;
  config.debug.modules_mask = DebugFeatureMask(DebugFeature::kMorphology) |
                              DebugFeatureMask(DebugFeature::kInferLogs);

  EXPECT_TRUE(pellet::utils::IsAnyDebugEnabled(config));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kMorphology));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kInferLogs));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kCaptureLogs));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kPipelineStats));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kThreadStatus));
}

TEST(DebugUtilsTest, LevelTwoEnablesAllFeaturesWhenMaskIsZero) {
  pellet::PelletConfig config;
  config.debug.enable = true;
  config.debug.level = 2;
  config.debug.modules_mask = 0U;

  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kCaptureLogs));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kMorphology));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kPipelineStats));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kThreadStatus));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kShowWindow));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kInferLogs));
}

}  // namespace
