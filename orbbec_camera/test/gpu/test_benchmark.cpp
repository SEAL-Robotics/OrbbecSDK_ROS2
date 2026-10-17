// Copyright 2026 SEAL Robotics
//
// CPU time per 1280x800 frame of the SDK's post-processing against the GPU
// pipeline's, measured as process CPU time so the SDK's worker threads
// count. It asserts only that the GPU path costs less; the numbers it
// prints are what the driver's CPU saving is estimated from. The SDK's
// noise filter is missing from it: it cannot run without a camera.

#include <gtest/gtest.h>
#include <time.h>

#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "jpeg_test_util.h"
#include "libobsensor/ObSensor.hpp"
#include "orbbec_camera/gpu/pipelines.h"
#include "orbbec_camera/gpu/sdk_algorithms.h"
#include "synthetic_frames.h"

namespace orbbec_camera {
namespace gpu {
namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 800;
constexpr int kFrames = 20;

double cpuSeconds() {
  timespec ts{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + ts.tv_nsec * 1e-9;
}

double wallSeconds() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + ts.tv_nsec * 1e-9;
}

struct Cost {
  double cpu_ms;
  double wall_ms;
};

Cost measure(const std::function<void()> &frame) {
  frame();
  const double cpu0 = cpuSeconds(), wall0 = wallSeconds();
  for (int i = 0; i < kFrames; ++i) {
    frame();
  }
  return {(cpuSeconds() - cpu0) * 1e3 / kFrames, (wallSeconds() - wall0) * 1e3 / kFrames};
}

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

void report(const char *what, const Cost &cost) {
  std::printf("[ cost ] %-58s %7.2f ms CPU  %7.2f ms wall per frame\n", what, cost.cpu_ms,
              cost.wall_ms);
}

TEST(Benchmark, GpuPipelineCostsLessCpuThanTheSdk) {
  const auto intrinsic = test::makeIntrinsic(kWidth, kHeight);
  const auto distortion = test::makeColorDistortion();
  const auto source = test::makeColor(kWidth, kHeight, 3, static_cast<size_t>(kWidth) * 3);
  const auto jpeg = test::encodeJpeg(source, kWidth, kHeight, 2, 1, 90);

  // The SDK: MJPG decode and undistortion in its UnDistortionFilter.
  auto mjpg_profile = makeProfile(OB_STREAM_COLOR, OB_FORMAT_MJPG, intrinsic, distortion);
  ob::UnDistortionFilter undistortion(OB_STREAM_COLOR);
  const Cost sdk_color = measure([&] {
    auto frame = ob::FrameFactory::createFrameFromStreamProfile(mjpg_profile);
    frame->updateData(jpeg.data(), static_cast<uint32_t>(jpeg.size()));
    ASSERT_TRUE(undistortion.process(frame));
  });
  report("SDK MJPG decode + undistortion", sdk_color);

  std::vector<uint8_t> rgb(source.size()), encoded;
  std::string error;
  const Cost cpu_jpeg = measure([&] {
    ASSERT_TRUE(encodeRgbJpegCpu(rgb.data(), kWidth, kHeight, 75, &encoded, &error));
  });
  report("CPU JPEG encode of the rectified frame (image_jpeg's work)", cpu_jpeg);

  std::string description;
  auto color = ColorPipeline::create(true, true, &description);
  ASSERT_TRUE(color->configure(buildUndistortLut(intrinsic, distortion), &error)) << error;
  const Cost gpu_color = measure([&] {
    ASSERT_TRUE(color->process(jpeg.data(), jpeg.size(), rgb.data(), &encoded, 75, &error))
        << error;
  });
  report(("pipeline: " + description).substr(0, 58).c_str(), gpu_color);

  // The SDK's software alignment against the depth pipeline's filter and
  // alignment.
  D2CConfig config;
  config.depth_intrinsic = test::makeIntrinsic(kWidth, kHeight, 0.98f);
  config.color_intrinsic = intrinsic;
  config.depth_to_color = test::makeDepthToColor();
  config.gap_fill_copy = true;
  const auto depth = test::makeDepth(kWidth, kHeight);
  auto depth_profile =
      makeProfile(OB_STREAM_DEPTH, OB_FORMAT_Y16, config.depth_intrinsic, OBCameraDistortion{});
  auto rgb_profile = makeProfile(OB_STREAM_COLOR, OB_FORMAT_RGB, intrinsic, distortion);
  depth_profile->bindExtrinsicTo(rgb_profile, config.depth_to_color);
  ob::Align align(OB_STREAM_COLOR);
  const Cost sdk_depth = measure([&] {
    auto depth_frame = ob::FrameFactory::createFrameFromStreamProfile(depth_profile);
    depth_frame->updateData(reinterpret_cast<const uint8_t *>(depth.data()),
                            static_cast<uint32_t>(depth.size() * sizeof(uint16_t)));
    auto color_frame = ob::FrameFactory::createFrameFromStreamProfile(rgb_profile);
    auto frame_set = ob::FrameFactory::createFrameSet();
    frame_set->pushFrame(depth_frame);
    frame_set->pushFrame(color_frame);
    ASSERT_TRUE(align.process(frame_set));
  });
  report("SDK depth-to-colour alignment", sdk_depth);

  auto depth_pipeline = DepthPipeline::create(true, &description);
  depth_pipeline->setSpeckle({256, 200});
  ASSERT_TRUE(depth_pipeline->setAlignment(config, &error)) << error;
  std::vector<uint16_t> aligned(depth.size());
  const Cost gpu_depth = measure([&] {
    ASSERT_TRUE(depth_pipeline->process(depth.data(), kWidth, kHeight, true, nullptr,
                                        aligned.data(), &error))
        << error;
  });
  report(("pipeline: " + description + ", speckle included").substr(0, 58).c_str(), gpu_depth);

  if (color->backend() != ColorPipeline::Backend::kCpu) {
    EXPECT_LT(gpu_color.cpu_ms, sdk_color.cpu_ms + cpu_jpeg.cpu_ms);
  }
  if (depth_pipeline->usesGpu()) {
    EXPECT_LT(gpu_depth.cpu_ms, sdk_depth.cpu_ms);
  }
}

}  // namespace
}  // namespace gpu
}  // namespace orbbec_camera
