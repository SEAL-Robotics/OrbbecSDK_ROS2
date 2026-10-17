// Copyright 2026 SEAL Robotics
//
// The pipelines as the driver uses them: the best colour backend this
// machine offers (NVJPG and VIC on a Jetson) against the CPU reference of
// decode then remap, and the depth backend against the CPU references.

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <string>
#include <vector>

#include "jpeg_test_util.h"
#include "orbbec_camera/gpu/image_metrics.h"
#include "orbbec_camera/gpu/pipelines.h"
#include "orbbec_camera/gpu/sdk_algorithms.h"
#include "synthetic_frames.h"

namespace orbbec_camera {
namespace gpu {
namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 800;

// 4:2:2, as the camera sends it.
std::vector<uint8_t> cameraJpeg(const std::vector<uint8_t> &rgb) {
  return test::encodeJpeg(rgb, kWidth, kHeight, 2, 1, 95);
}

TEST(ColorPipeline, RectifiedRgbAndJpegMatchTheReference) {
  const auto source = test::makeColor(kWidth, kHeight, 3, static_cast<size_t>(kWidth) * 3);
  const auto jpeg = cameraJpeg(source);
  const auto lut = buildUndistortLut(test::makeIntrinsic(kWidth, kHeight),
                                     test::makeColorDistortion());

  std::vector<uint8_t> decoded;
  std::string error;
  ASSERT_TRUE(decodeJpegRgbCpu(jpeg.data(), jpeg.size(), kWidth, kHeight, &decoded, &error));
  std::vector<uint8_t> reference(decoded.size());
  remapRgbCpu(decoded.data(), 3, static_cast<size_t>(kWidth) * 3, reference.data(), lut);

  std::string description;
  auto pipeline = ColorPipeline::create(true, true, &description);
  std::printf("[ backend ] %s\n", description.c_str());
  ASSERT_TRUE(pipeline->configure(lut, &error)) << error;

  std::vector<uint8_t> rgb(reference.size());
  std::vector<uint8_t> encoded;
  for (int frame = 0; frame < 3; ++frame) {
    ASSERT_TRUE(pipeline->process(jpeg.data(), jpeg.size(), rgb.data(), &encoded, 75, &error))
        << error;
  }
  const auto rgb_diff = compareColor(reference.data(), rgb.data(), rgb.size());
  std::printf("[ colour ] %s vs CPU decode+remap: PSNR %.2f dB, max %d\n",
              backendName(pipeline->backend()), rgb_diff.psnr_db, rgb_diff.max_abs_diff);
  if (pipeline->backend() == ColorPipeline::Backend::kNvHardware) {
    // NVJPG's inverse DCT is not libjpeg's: close, not equal.
    EXPECT_GT(rgb_diff.psnr_db, 45.0);
    EXPECT_LE(rgb_diff.max_abs_diff, 8);
  } else {
    EXPECT_EQ(rgb_diff.max_abs_diff, 0);
  }

  // The JPEG is held to what libjpeg makes of the same rectified frame at
  // the same quality.
  std::vector<uint8_t> roundtrip, cpu_jpeg, cpu_roundtrip;
  ASSERT_TRUE(decodeJpegRgbCpu(encoded.data(), encoded.size(), kWidth, kHeight, &roundtrip,
                               &error))
      << error;
  ASSERT_TRUE(encodeRgbJpegCpu(rgb.data(), kWidth, kHeight, 75, &cpu_jpeg, &error)) << error;
  ASSERT_TRUE(decodeJpegRgbCpu(cpu_jpeg.data(), cpu_jpeg.size(), kWidth, kHeight, &cpu_roundtrip,
                               &error))
      << error;
  const auto jpeg_diff = compareColor(rgb.data(), roundtrip.data(), rgb.size());
  const auto cpu_diff = compareColor(rgb.data(), cpu_roundtrip.data(), rgb.size());
  std::printf("[ jpeg ] %zu bytes at quality 75, PSNR %.2f dB against the rectified frame "
              "(libjpeg: %zu bytes, %.2f dB)\n",
              encoded.size(), jpeg_diff.psnr_db, cpu_jpeg.size(), cpu_diff.psnr_db);
  EXPECT_GT(jpeg_diff.psnr_db, cpu_diff.psnr_db - 1.5);
  EXPECT_LT(encoded.size(), cpu_jpeg.size() * 3 / 2);
}

TEST(ColorPipeline, CompressedOnlyNeedsNoRgbBuffer) {
  const auto source = test::makeColor(kWidth, kHeight, 3, static_cast<size_t>(kWidth) * 3);
  const auto jpeg = cameraJpeg(source);
  const auto lut = buildUndistortLut(test::makeIntrinsic(kWidth, kHeight),
                                     test::makeColorDistortion());
  auto pipeline = ColorPipeline::create(true, true, nullptr);
  std::string error;
  ASSERT_TRUE(pipeline->configure(lut, &error)) << error;
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(pipeline->process(jpeg.data(), jpeg.size(), nullptr, &encoded, 75, &error)) << error;
  EXPECT_GT(encoded.size(), 1000u);
  EXPECT_EQ(encoded[0], 0xFF);
  EXPECT_EQ(encoded[1], 0xD8);
}

TEST(ColorPipeline, CorruptJpegFailsTheFrameNotTheProcess) {
  const auto lut = buildUndistortLut(test::makeIntrinsic(kWidth, kHeight),
                                     test::makeColorDistortion());
  auto pipeline = ColorPipeline::create(true, true, nullptr);
  std::string error;
  ASSERT_TRUE(pipeline->configure(lut, &error)) << error;
  const auto source = test::makeColor(kWidth, kHeight, 3, static_cast<size_t>(kWidth) * 3);
  auto jpeg = cameraJpeg(source);
  std::vector<uint8_t> corrupt(jpeg.begin(), jpeg.begin() + 600);
  corrupt[0] = 0x00;  // no start-of-image marker
  std::vector<uint8_t> rgb(static_cast<size_t>(kWidth) * kHeight * 3);
  EXPECT_FALSE(pipeline->process(corrupt.data(), corrupt.size(), rgb.data(), nullptr, 75,
                                 &error));
  // And the next good frame still decodes.
  EXPECT_TRUE(pipeline->process(jpeg.data(), jpeg.size(), rgb.data(), nullptr, 75, &error))
      << error;
}

TEST(ColorPipeline, CpuBackendMatchesReferenceExactly) {
  const auto source = test::makeColor(320, 200, 3, 320 * 3);
  cv::Mat bgr;
  cv::cvtColor(cv::Mat(200, 320, CV_8UC3, const_cast<uint8_t *>(source.data())), bgr,
               cv::COLOR_RGB2BGR);
  std::vector<uint8_t> jpeg;
  cv::imencode(".jpg", bgr, jpeg, {cv::IMWRITE_JPEG_QUALITY, 90});
  const auto lut = buildUndistortLut(test::makeIntrinsic(320, 200), test::makeColorDistortion());
  auto pipeline = ColorPipeline::create(false, false, nullptr);
  ASSERT_EQ(pipeline->backend(), ColorPipeline::Backend::kCpu);
  std::string error;
  ASSERT_TRUE(pipeline->configure(lut, &error));
  std::vector<uint8_t> rgb(320 * 200 * 3), decoded, reference(rgb.size());
  ASSERT_TRUE(pipeline->process(jpeg.data(), jpeg.size(), rgb.data(), nullptr, 75, &error));
  ASSERT_TRUE(decodeJpegRgbCpu(jpeg.data(), jpeg.size(), 320, 200, &decoded, &error));
  remapRgbCpu(decoded.data(), 3, 320 * 3, reference.data(), lut);
  EXPECT_EQ(rgb, reference);
}

TEST(DepthPipeline, BestBackendMatchesCpuReferences) {
  D2CConfig config;
  config.depth_intrinsic = test::makeIntrinsic(kWidth, kHeight, 0.98f);
  config.color_intrinsic = test::makeIntrinsic(kWidth, kHeight);
  config.depth_to_color = test::makeDepthToColor();
  config.gap_fill_copy = true;
  const SpeckleParams speckle{256, 200};
  const auto depth = test::makeDepth(kWidth, kHeight);

  std::vector<uint16_t> filtered(depth.size()), aligned(depth.size());
  speckleFilterCpu(depth.data(), filtered.data(), kWidth, kHeight, speckle);
  alignDepthToColorCpu(filtered.data(), aligned.data(), config, buildD2CTables(config));

  for (const bool cuda : {true, false}) {
    std::string description, error;
    auto pipeline = DepthPipeline::create(cuda, &description);
    std::printf("[ backend ] %s\n", description.c_str());
    pipeline->setSpeckle(speckle);
    ASSERT_TRUE(pipeline->setAlignment(config, &error)) << error;
    std::vector<uint16_t> out_filtered(depth.size()), out_aligned(depth.size());
    ASSERT_TRUE(pipeline->process(depth.data(), kWidth, kHeight, true, out_filtered.data(),
                                  out_aligned.data(), &error))
        << error;
    EXPECT_EQ(filtered, out_filtered);
    EXPECT_EQ(aligned, out_aligned);
  }
}

}  // namespace
}  // namespace gpu
}  // namespace orbbec_camera
