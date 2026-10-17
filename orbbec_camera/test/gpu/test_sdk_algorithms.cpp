// Copyright 2026 SEAL Robotics
//
// The CPU ports against the SDK itself: its public UnDistortionFilter and
// Align run here on the same synthetic frames, so a port that drifts from
// the SDK's arithmetic fails on this machine's SDK build. The speckle filter
// has no public SDK counterpart and is pinned by known answers instead.

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "libobsensor/ObSensor.hpp"
#include "orbbec_camera/gpu/image_metrics.h"
#include "orbbec_camera/gpu/sdk_algorithms.h"
#include "synthetic_frames.h"

namespace orbbec_camera {
namespace gpu {
namespace {

std::shared_ptr<ob::VideoStreamProfile> makeProfile(OBStreamType type, OBFormat format,
                                                    const OBCameraIntrinsic &intrinsic,
                                                    const OBCameraDistortion &distortion) {
  ob_error *error = nullptr;
  ob_stream_profile *raw = ob_create_video_stream_profile(
      type, format, static_cast<uint32_t>(intrinsic.width), static_cast<uint32_t>(intrinsic.height),
      15, &error);
  ob::Error::handle(&error);
  auto profile = ob::StreamProfileFactory::create(raw)->as<ob::VideoStreamProfile>();
  profile->setIntrinsic(intrinsic);
  profile->setDistortion(distortion);
  return profile;
}

std::shared_ptr<ob::Frame> makeFrame(const std::shared_ptr<ob::VideoStreamProfile> &profile,
                                     const void *data, size_t bytes) {
  auto frame = ob::FrameFactory::createFrameFromStreamProfile(profile);
  EXPECT_GE(frame->getDataSize(), bytes);
  std::memcpy(frame->getData(), data, bytes);
  return frame;
}

TEST(SdkUndistortion, CpuPortMatchesSdkFilterBitForBit) {
  const int width = 1280, height = 800;
  const auto intrinsic = test::makeIntrinsic(width, height);
  const auto distortion = test::makeColorDistortion();
  const auto rgb = test::makeColor(width, height, 3, static_cast<size_t>(width) * 3);

  auto profile = makeProfile(OB_STREAM_COLOR, OB_FORMAT_RGB, intrinsic, distortion);
  auto frame = makeFrame(profile, rgb.data(), rgb.size());
  ob::UnDistortionFilter filter(OB_STREAM_COLOR);
  auto sdk = filter.process(frame);
  ASSERT_TRUE(sdk);
  auto sdk_profile = sdk->getStreamProfile()->as<ob::VideoStreamProfile>();
  const OBCameraDistortion out_distortion = sdk_profile->getDistortion();
  EXPECT_EQ(out_distortion.k1, 0.0f);
  EXPECT_EQ(out_distortion.model, distortion.model);

  std::vector<uint8_t> ours(rgb.size());
  remapRgbCpu(rgb.data(), 3, static_cast<size_t>(width) * 3, ours.data(),
              buildUndistortLut(intrinsic, distortion));
  const auto diff = compareColor(sdk->getData(), ours.data(),
                                 ours.size());
  EXPECT_EQ(diff.max_abs_diff, 0) << "PSNR " << diff.psnr_db;
}

TEST(SdkUndistortion, FourChannelSourceGivesTheSameRgb) {
  const int width = 320, height = 200;
  const auto intrinsic = test::makeIntrinsic(width, height);
  const auto lut = buildUndistortLut(intrinsic, test::makeColorDistortion());
  const size_t pitch3 = static_cast<size_t>(width) * 3;
  const size_t pitch4 = static_cast<size_t>(width) * 4 + 64;
  const auto rgba = test::makeColor(width, height, 4, pitch4);
  std::vector<uint8_t> rgb(pitch3 * height);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      std::memcpy(&rgb[v * pitch3 + u * 3], &rgba[v * pitch4 + u * 4], 3);
    }
  }
  std::vector<uint8_t> from3(rgb.size()), from4(rgb.size());
  remapRgbCpu(rgb.data(), 3, pitch3, from3.data(), lut);
  remapRgbCpu(rgba.data(), 4, pitch4, from4.data(), lut);
  EXPECT_EQ(from3, from4);
}

struct AlignCase {
  int depth_width;
  int depth_height;
  bool depth_distortion;
};

class SdkAlign : public ::testing::TestWithParam<AlignCase> {};

TEST_P(SdkAlign, CpuPortMatchesSdkAlignBitForBit) {
  const AlignCase c = GetParam();
  const int color_width = 1280, color_height = 800;
  D2CConfig config;
  config.depth_intrinsic = test::makeIntrinsic(c.depth_width, c.depth_height, 0.98f);
  if (c.depth_distortion) {
    config.depth_distortion.k1 = 0.012f;
    config.depth_distortion.p2 = -0.0004f;
    config.depth_distortion.model = OB_DISTORTION_BROWN_CONRADY;
  }
  config.color_intrinsic = test::makeIntrinsic(color_width, color_height);
  config.depth_to_color = test::makeDepthToColor();
  config.depth_unit_mm = 1.0f;
  config.max_valid_depth = 65535;
  config.gap_fill_copy = sdkGapFillCopy(config.depth_intrinsic, config.color_intrinsic);

  const auto depth = test::makeDepth(c.depth_width, c.depth_height);
  const auto color = test::makeColor(color_width, color_height, 3,
                                     static_cast<size_t>(color_width) * 3);

  auto depth_profile = makeProfile(OB_STREAM_DEPTH, OB_FORMAT_Y16, config.depth_intrinsic,
                                   config.depth_distortion);
  auto color_profile = makeProfile(OB_STREAM_COLOR, OB_FORMAT_RGB, config.color_intrinsic,
                                   test::makeColorDistortion());
  depth_profile->bindExtrinsicTo(color_profile, config.depth_to_color);
  auto depth_frame = makeFrame(depth_profile, depth.data(), depth.size() * sizeof(uint16_t));
  depth_frame->as<ob::DepthFrame>()->setValueScale(config.depth_unit_mm);
  auto color_frame = makeFrame(color_profile, color.data(), color.size());
  auto frame_set = ob::FrameFactory::createFrameSet();
  frame_set->pushFrame(depth_frame);
  frame_set->pushFrame(color_frame);

  ob::Align align(OB_STREAM_COLOR);
  auto result = align.process(frame_set);
  ASSERT_TRUE(result);
  auto aligned = result->as<ob::FrameSet>()->getFrame(OB_FRAME_DEPTH);
  ASSERT_TRUE(aligned);
  auto aligned_video = aligned->as<ob::VideoFrame>();
  ASSERT_EQ(static_cast<int>(aligned_video->getWidth()), color_width);
  ASSERT_EQ(static_cast<int>(aligned_video->getHeight()), color_height);

  std::vector<uint16_t> ours(static_cast<size_t>(color_width) * color_height);
  alignDepthToColorCpu(depth.data(), ours.data(), config, buildD2CTables(config));
  const auto diff = compareDepth(reinterpret_cast<const uint16_t *>(aligned->getData()), ours.data(),
                                 ours.size());
  EXPECT_GT(diff.valid_a, ours.size() / 2);
  EXPECT_EQ(diff.valid_a, diff.valid_b);
  EXPECT_EQ(diff.valid_both, diff.valid_either);
  EXPECT_EQ(diff.differing, 0u) << "max diff " << diff.max_abs_diff;
}

INSTANTIATE_TEST_SUITE_P(GapFill, SdkAlign,
                         ::testing::Values(AlignCase{1280, 800, false}, AlignCase{1280, 800, true},
                                           AlignCase{640, 400, false}));

TEST(SdkAlign, GapFillFollowsResolutionRatio) {
  const auto color = test::makeIntrinsic(1280, 800);
  EXPECT_TRUE(sdkGapFillCopy(test::makeIntrinsic(1280, 800, 0.98f), color));
  EXPECT_FALSE(sdkGapFillCopy(test::makeIntrinsic(640, 400, 0.98f), color));
}

TEST(SdkAlign, DepthAtOrAboveMaxValidIsDropped) {
  const int width = 64, height = 40;
  D2CConfig config;
  config.depth_intrinsic = test::makeIntrinsic(width, height);
  config.color_intrinsic = test::makeIntrinsic(width, height);
  config.depth_to_color.rot[0] = config.depth_to_color.rot[4] = config.depth_to_color.rot[8] = 1;
  config.max_valid_depth = 1000;
  std::vector<uint16_t> depth(width * height, 999);
  depth[10 * width + 10] = 1000;
  std::vector<uint16_t> out(depth.size());
  alignDepthToColorCpu(depth.data(), out.data(), config, buildD2CTables(config));
  EXPECT_EQ(out[20 * width + 20], 999);
  for (uint16_t value : out) {
    EXPECT_LT(value, 1000);
  }
}

// ---------------------------------------------------------------------------

std::vector<uint16_t> filled(int width, int height, uint16_t value) {
  return std::vector<uint16_t>(static_cast<size_t>(width) * height, value);
}

void paint(std::vector<uint16_t> &depth, int width, int x0, int y0, int w, int h,
           uint16_t value) {
  for (int v = y0; v < y0 + h; ++v) {
    for (int u = x0; u < x0 + w; ++u) {
      depth[static_cast<size_t>(v) * width + u] = value;
    }
  }
}

TEST(SpeckleFilter, RemovesRegionsUpToMaxSizeOnly) {
  const int width = 40, height = 30;
  auto depth = filled(width, height, 1000);
  paint(depth, width, 2, 2, 4, 5, 3000);    // 20 pixels
  paint(depth, width, 20, 10, 3, 7, 3000);  // 21 pixels
  std::vector<uint16_t> out(depth.size());
  speckleFilterCpu(depth.data(), out.data(), width, height, {256, 20});
  EXPECT_EQ(out[3 * width + 3], 0);
  EXPECT_EQ(out[11 * width + 21], 3000);
  EXPECT_EQ(out[0], 1000);
}

TEST(SpeckleFilter, NeighboursWithinMaxDiffJoinOneRegion) {
  const int width = 40, height = 30;
  auto depth = filled(width, height, 0);
  // A ramp of 30 pixels whose steps are exactly max_diff joins up; the same
  // ramp one unit steeper falls apart into single pixels.
  for (int u = 0; u < 30; ++u) {
    depth[5 * width + u] = static_cast<uint16_t>(500 + 256 * u);
    depth[20 * width + u] = static_cast<uint16_t>(500 + 257 * u);
  }
  std::vector<uint16_t> out(depth.size());
  speckleFilterCpu(depth.data(), out.data(), width, height, {256, 20});
  for (int u = 0; u < 30; ++u) {
    EXPECT_EQ(out[5 * width + u], depth[5 * width + u]);
    EXPECT_EQ(out[20 * width + u], 0);
  }
}

TEST(SpeckleFilter, DiagonalNeighboursDoNotConnect) {
  const int width = 10, height = 10;
  auto depth = filled(width, height, 0);
  for (int i = 0; i < 10; ++i) {
    depth[i * width + i] = 1000;
  }
  std::vector<uint16_t> out(depth.size());
  speckleFilterCpu(depth.data(), out.data(), width, height, {256, 1});
  for (uint16_t value : out) {
    EXPECT_EQ(value, 0);
  }
}

TEST(SpeckleFilter, ZeroDepthIsNeverARegion) {
  const int width = 16, height = 16;
  auto depth = filled(width, height, 0);
  paint(depth, width, 4, 4, 2, 2, 7);
  std::vector<uint16_t> out(depth.size(), 1);
  speckleFilterCpu(depth.data(), out.data(), width, height, {256, 0});
  EXPECT_EQ(out[4 * width + 4], 7);
  EXPECT_EQ(out[0], 0);
}

}  // namespace
}  // namespace gpu
}  // namespace orbbec_camera
