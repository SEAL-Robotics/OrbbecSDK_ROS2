// Copyright 2026 SEAL Robotics
//
// GpuImagePipeline on SDK frames built here, as the driver hands them over:
// its outputs carry the SDK's own output profiles and the input's frame
// info, their pixels equal the SDK's (alignment) or the CPU reference's,
// and comparison mode publishes what it measured. CMakeLists.txt runs this
// on a private ROS domain with localhost-only discovery.

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "jpeg_test_util.h"
#include "libobsensor/ObSensor.hpp"
#include "orbbec_camera/gpu/image_metrics.h"
#include "orbbec_camera/gpu/sdk_algorithms.h"
#include "orbbec_camera/gpu_image_pipeline.h"
#include "synthetic_frames.h"

namespace orbbec_camera {
namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 800;

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

class GpuImagePipelineTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override {
    std::string reason;
    if (!GpuImagePipeline::available(&reason)) {
      GTEST_SKIP() << reason;
    }
    node_ = std::make_shared<rclcpp::Node>("gpu_image_pipeline_test");
    depth_intrinsic_ = gpu::test::makeIntrinsic(kWidth, kHeight, 0.98f);
    color_intrinsic_ = gpu::test::makeIntrinsic(kWidth, kHeight);
    depth_profile_ = makeProfile(OB_STREAM_DEPTH, OB_FORMAT_Y16, depth_intrinsic_, {});
    color_profile_ = makeProfile(OB_STREAM_COLOR, OB_FORMAT_MJPG, color_intrinsic_,
                                 gpu::test::makeColorDistortion());
    depth_profile_->bindExtrinsicTo(color_profile_, gpu::test::makeDepthToColor());
  }

  std::shared_ptr<ob::Frame> depthFrame(const std::vector<uint16_t> &depth, uint64_t stamp) {
    auto frame = ob::FrameFactory::createFrameFromStreamProfile(depth_profile_);
    frame->updateData(reinterpret_cast<const uint8_t *>(depth.data()),
                      static_cast<uint32_t>(depth.size() * sizeof(uint16_t)));
    frame->as<ob::DepthFrame>()->setValueScale(1.0f);
    ob::FrameHelper::setFrameDeviceTimestampUs(frame, stamp);
    return frame;
  }

  std::shared_ptr<ob::Frame> colorFrame(const std::vector<uint8_t> &jpeg, uint64_t stamp) {
    auto frame = ob::FrameFactory::createFrameFromStreamProfile(color_profile_);
    // The camera pads its JPEG with zeros, which the pipeline must ignore.
    std::vector<uint8_t> padded(jpeg);
    padded.resize(jpeg.size() + 64, 0);
    frame->updateData(padded.data(), static_cast<uint32_t>(padded.size()));
    ob::FrameHelper::setFrameDeviceTimestampUs(frame, stamp);
    return frame;
  }

  std::shared_ptr<rclcpp::Node> node_;
  OBCameraIntrinsic depth_intrinsic_{};
  OBCameraIntrinsic color_intrinsic_{};
  std::shared_ptr<ob::VideoStreamProfile> depth_profile_;
  std::shared_ptr<ob::VideoStreamProfile> color_profile_;
};

TEST_F(GpuImagePipelineTest, DepthMatchesSdkAlignmentWithItsProfileAndStamp) {
  GpuImagePipeline::Options options;
  options.speckle = {256, 200};
  GpuImagePipeline pipeline(node_.get(), options);
  const auto depth = gpu::test::makeDepth(kWidth, kHeight);
  const auto jpeg = gpu::test::encodeJpeg(
      gpu::test::makeColor(kWidth, kHeight, 3, kWidth * 3), kWidth, kHeight, 2, 1, 90);
  ob::Align sdk_align(OB_STREAM_COLOR);

  for (uint64_t stamp : {1000000ull, 1066666ull}) {
    auto raw = depthFrame(depth, stamp);
    auto color = colorFrame(jpeg, stamp);
    pipeline.startDepthFrame();
    auto filtered = pipeline.filterDepth(raw, true);
    ASSERT_TRUE(filtered);
    std::vector<uint16_t> expected(depth.size());
    gpu::speckleFilterCpu(depth.data(), expected.data(), kWidth, kHeight, {256, 200});
    EXPECT_EQ(std::memcmp(filtered->getData(), expected.data(), expected.size() * 2), 0);
    EXPECT_EQ(filtered->getTimeStampUs(), stamp);

    auto aligned = pipeline.alignDepth(filtered, color, sdk_align, true);
    ASSERT_TRUE(aligned);
    auto frame_set = ob::FrameFactory::createFrameSet();
    frame_set->pushFrame(filtered);
    frame_set->pushFrame(color);
    auto sdk = sdk_align.process(frame_set)->as<ob::FrameSet>()->getFrame(OB_FRAME_DEPTH);
    ASSERT_EQ(aligned->getDataSize(), sdk->getDataSize());
    const auto diff =
        gpu::compareDepth(reinterpret_cast<const uint16_t *>(sdk->getData()),
                          reinterpret_cast<const uint16_t *>(aligned->getData()), kWidth * kHeight);
    EXPECT_EQ(diff.differing, 0u);
    EXPECT_EQ(diff.validIoU(), 1.0);
    EXPECT_EQ(aligned->getTimeStampUs(), stamp);
    EXPECT_FLOAT_EQ(aligned->as<ob::DepthFrame>()->getValueScale(), 1.0f);
    auto profile = aligned->getStreamProfile()->as<ob::VideoStreamProfile>();
    auto sdk_profile = sdk->getStreamProfile()->as<ob::VideoStreamProfile>();
    EXPECT_EQ(profile->getWidth(), sdk_profile->getWidth());
    const auto intrinsic = profile->getIntrinsic();
    const auto sdk_intrinsic = sdk_profile->getIntrinsic();
    EXPECT_EQ(std::memcmp(&intrinsic, &sdk_intrinsic, sizeof(OBCameraIntrinsic)), 0);
    const auto distortion = profile->getDistortion();
    const auto sdk_distortion = sdk_profile->getDistortion();
    EXPECT_EQ(std::memcmp(&distortion, &sdk_distortion, sizeof(OBCameraDistortion)), 0);
  }
}

TEST_F(GpuImagePipelineTest, ColourIsRectifiedWithSdkProfileAndEncoded) {
  GpuImagePipeline::Options options;
  GpuImagePipeline pipeline(node_.get(), options);
  const auto source = gpu::test::makeColor(kWidth, kHeight, 3, kWidth * 3);
  const auto jpeg = gpu::test::encodeJpeg(source, kWidth, kHeight, 2, 1, 90);
  auto color = colorFrame(jpeg, 4242);

  std::shared_ptr<ob::Frame> rectified;
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(pipeline.processColor(color, true, true, &rectified, &encoded));
  ASSERT_TRUE(rectified);
  EXPECT_EQ(rectified->getFormat(), OB_FORMAT_RGB);
  EXPECT_EQ(rectified->getTimeStampUs(), 4242u);
  EXPECT_GT(encoded.size(), 1000u);

  ob::UnDistortionFilter sdk_filter(OB_STREAM_COLOR);
  auto sdk = sdk_filter.process(color);
  ASSERT_EQ(sdk->getDataSize(), rectified->getDataSize());
  const auto diff = gpu::compareColor(sdk->getData(), rectified->getData(), sdk->getDataSize());
  // Exact with the CPU decode; NVJPG's inverse DCT is not libjpeg's fast
  // one the SDK uses.
  EXPECT_GT(diff.psnr_db, 40.0) << "max " << diff.max_abs_diff;
  EXPECT_LE(diff.max_abs_diff, 10);
  const auto distortion = rectified->getStreamProfile()->as<ob::VideoStreamProfile>()->getDistortion();
  EXPECT_EQ(distortion.k1, 0.0f);

  // Compressed only: no RGB is computed, the JPEG still is.
  encoded.clear();
  ASSERT_TRUE(pipeline.processColor(color, false, true, &rectified, &encoded));
  EXPECT_GT(encoded.size(), 1000u);
}

TEST_F(GpuImagePipelineTest, UncomputedColourIsBlackNeverStale) {
  GpuImagePipeline pipeline(node_.get(), GpuImagePipeline::Options{});
  const auto jpeg = gpu::test::encodeJpeg(
      gpu::test::makeColor(kWidth, kHeight, 3, kWidth * 3), kWidth, kHeight, 2, 1, 90);
  std::shared_ptr<ob::Frame> rectified;
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(pipeline.processColor(colorFrame(jpeg, 7), false, false, &rectified, &encoded));
  const uint8_t *data = rectified->getData();
  EXPECT_TRUE(std::all_of(data, data + rectified->getDataSize(), [](uint8_t v) { return v == 0; }));
  EXPECT_TRUE(encoded.empty());
}

TEST_F(GpuImagePipelineTest, ColourWithoutDistortionIsLeftToTheSdk) {
  color_profile_ = makeProfile(OB_STREAM_COLOR, OB_FORMAT_MJPG, color_intrinsic_, {});
  GpuImagePipeline pipeline(node_.get(), GpuImagePipeline::Options{});
  const auto jpeg = gpu::test::encodeJpeg(
      gpu::test::makeColor(kWidth, kHeight, 3, kWidth * 3), kWidth, kHeight, 2, 1, 90);
  auto color = colorFrame(jpeg, 7);
  std::shared_ptr<ob::Frame> rectified;
  std::vector<uint8_t> encoded;
  EXPECT_FALSE(pipeline.processColor(color, true, true, &rectified, &encoded));
  // The SDK passes an undistorted stream through as MJPG, which the driver
  // then decodes on its usual path.
  auto sdk = pipeline.sdkRectify(color);
  ASSERT_TRUE(sdk);
  EXPECT_EQ(sdk->getFormat(), OB_FORMAT_MJPG);
}

TEST_F(GpuImagePipelineTest, ComparisonModePublishesEveryStage) {
  GpuImagePipeline::Options options;
  options.compare = true;
  options.speckle = {256, 200};
  GpuImagePipeline pipeline(node_.get(), options);
  std::vector<std::string> stages;
  auto sub = node_->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
      "gpu_pipeline/comparison", 10, [&](diagnostic_msgs::msg::DiagnosticArray::SharedPtr msg) {
        for (const auto &status : msg->status) {
          stages.push_back(status.name);
        }
      });
  const auto depth = gpu::test::makeDepth(kWidth, kHeight);
  const auto jpeg = gpu::test::encodeJpeg(
      gpu::test::makeColor(kWidth, kHeight, 3, kWidth * 3), kWidth, kHeight, 2, 1, 90);
  ob::Align sdk_align(OB_STREAM_COLOR);
  auto color = colorFrame(jpeg, 1);
  pipeline.startDepthFrame();
  auto filtered = pipeline.filterDepth(depthFrame(depth, 1), false);
  ASSERT_TRUE(pipeline.alignDepth(filtered, color, sdk_align, false));
  std::shared_ptr<ob::Frame> rectified;
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(pipeline.processColor(color, false, false, &rectified, &encoded));
  for (int i = 0; i < 50 && stages.size() < 2; ++i) {
    rclcpp::spin_some(node_);
    rclcpp::sleep_for(std::chrono::milliseconds(20));
  }
  // The SDK's private noise filter needs a camera; without one the speckle
  // filter is compared through the alignment only.
  EXPECT_NE(std::find(stages.begin(), stages.end(), "gpu_pipeline/depth_alignment"),
            stages.end());
  EXPECT_NE(std::find(stages.begin(), stages.end(), "gpu_pipeline/color"), stages.end());
}

}  // namespace
}  // namespace orbbec_camera
