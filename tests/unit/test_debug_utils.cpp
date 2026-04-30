#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "pellet/config.hpp"
#include "pellet/utils/debug_utils.hpp"

namespace {

using pellet::utils::DebugFeature;

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
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kStats1s));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kInferLogs));
}

TEST(DebugUtilsTest, ModulesMaskHasHigherPriorityThanLevel) {
  pellet::PelletConfig config;
  config.debug.enable = true;
  config.debug.level = 2;
  config.debug.modules_mask = 4U;  // stats_1s

  EXPECT_TRUE(pellet::utils::IsAnyDebugEnabled(config));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kStats1s));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kInferLogs));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kCaptureLogs));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kPipelineStats));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kThreadStatus));
}

TEST(DebugUtilsTest, InvalidModulesMaskDisablesDebugFeatures) {
  pellet::PelletConfig config;
  config.debug.enable = true;
  config.debug.level = 2;
  config.debug.modules_mask = 9U;  // invalid selector

  EXPECT_FALSE(pellet::utils::IsAnyDebugEnabled(config));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kCaptureLogs));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kPipelineStats));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kThreadStatus));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kStats1s));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kInferLogs));
}

TEST(DebugUtilsTest, LevelTwoEnablesCoreFeaturesWithoutStats1s) {
  pellet::PelletConfig config;
  config.debug.enable = true;
  config.debug.level = 2;
  config.debug.modules_mask = 0U;

  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kCaptureLogs));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kPipelineStats));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kThreadStatus));
  EXPECT_FALSE(pellet::utils::IsDebugEnabled(config, DebugFeature::kStats1s));
  EXPECT_TRUE(pellet::utils::IsDebugEnabled(config, DebugFeature::kInferLogs));
}

TEST(DebugUtilsTest, RateLimitBlocksWithinInterval) {
  using namespace std::chrono_literals;
  EXPECT_TRUE(pellet::utils::ShouldLogRateLimited("debug_utils_test", "same_key", 50ms));
  EXPECT_FALSE(pellet::utils::ShouldLogRateLimited("debug_utils_test", "same_key", 50ms));
}

TEST(DebugUtilsTest, RateLimitAllowsAfterInterval) {
  using namespace std::chrono_literals;
  EXPECT_TRUE(pellet::utils::ShouldLogRateLimited("debug_utils_test", "recover_key", 10ms));
  EXPECT_FALSE(pellet::utils::ShouldLogRateLimited("debug_utils_test", "recover_key", 10ms));
  std::this_thread::sleep_for(20ms);
  EXPECT_TRUE(pellet::utils::ShouldLogRateLimited("debug_utils_test", "recover_key", 10ms));
}

TEST(DebugUtilsTest, RateLimitIsIndependentAcrossKeys) {
  using namespace std::chrono_literals;
  EXPECT_TRUE(pellet::utils::ShouldLogRateLimited("debug_utils_test", "key_a", 100ms));
  EXPECT_TRUE(pellet::utils::ShouldLogRateLimited("debug_utils_test", "key_b", 100ms));
}

}  // namespace
