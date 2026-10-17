// Copyright 2026 SEAL Robotics
//
// Every CUDA stage against its CPU reference, pixel for pixel, on frames the
// size of the 338Le's. Skipped on a machine without a CUDA device.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <cstring>
#include <string>
#include <vector>

#include "orbbec_camera/gpu/color_math.h"
#include "orbbec_camera/gpu/cuda_stages.h"
#include "orbbec_camera/gpu/image_metrics.h"
#include "orbbec_camera/gpu/sdk_algorithms.h"
#include "synthetic_frames.h"

namespace orbbec_camera {
namespace gpu {
namespace {

class CudaStages : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string reason;
    if (!cudaDeviceAvailable(&reason)) {
      GTEST_SKIP() << reason;
    }
  }
};

D2CConfig makeConfig(int depth_width, int depth_height) {
  D2CConfig config;
  config.depth_intrinsic = test::makeIntrinsic(depth_width, depth_height, 0.98f);
  config.color_intrinsic = test::makeIntrinsic(1280, 800);
  config.depth_to_color = test::makeDepthToColor();
  config.max_valid_depth = 26208;
  config.gap_fill_copy = sdkGapFillCopy(config.depth_intrinsic, config.color_intrinsic);
  return config;
}

TEST_F(CudaStages, SpeckleFilterMatchesCpu) {
  const int width = 1280, height = 800;
  for (const SpeckleParams &params : {SpeckleParams{256, 200}, SpeckleParams{256, 80},
                                     SpeckleParams{16, 200}, SpeckleParams{1000, 5}}) {
    const auto depth = test::makeDepth(width, height, 3 + params.max_size);
    std::vector<uint16_t> cpu(depth.size()), gpu(depth.size());
    speckleFilterCpu(depth.data(), cpu.data(), width, height, params);
    CudaDepthStage stage;
    stage.setSpeckle(params);
    std::string error;
    ASSERT_TRUE(stage.process(depth.data(), width, height, true, gpu.data(), nullptr, &error))
        << error;
    const auto diff = compareDepth(cpu.data(), gpu.data(), cpu.size());
    EXPECT_EQ(cpu, gpu) << "max_diff " << params.max_diff << " max_size " << params.max_size
                        << ": IoU " << diff.validIoU();
    // The synthetic blobs are there to be removed.
    EXPECT_LT(diff.valid_a, compareDepth(depth.data(), depth.data(), depth.size()).valid_a);
  }
}

TEST_F(CudaStages, SpeckleFilterSeesOneLongSnakeAsOneRegion) {
  // A one-pixel-wide serpentine joins pixels whose union-find roots sit far
  // apart: the worst case for a parallel labelling.
  const int width = 257, height = 101;
  std::vector<uint16_t> depth(static_cast<size_t>(width) * height, 0);
  for (int v = 0; v < height; v += 2) {
    for (int u = 0; u < width; ++u) {
      depth[static_cast<size_t>(v) * width + u] = 1500;
    }
    const int link = (v / 2) % 2 == 0 ? width - 1 : 0;
    if (v + 1 < height) {
      depth[static_cast<size_t>(v + 1) * width + link] = 1500;
    }
  }
  std::vector<uint16_t> cpu(depth.size()), gpu(depth.size());
  const SpeckleParams params{0, width * 51};
  speckleFilterCpu(depth.data(), cpu.data(), width, height, params);
  CudaDepthStage stage;
  stage.setSpeckle(params);
  std::string error;
  ASSERT_TRUE(stage.process(depth.data(), width, height, true, gpu.data(), nullptr, &error))
      << error;
  EXPECT_EQ(cpu, gpu);
  EXPECT_EQ(gpu[0], 1500);
}

TEST_F(CudaStages, AlignmentMatchesCpuForBothGapFills) {
  for (const auto &size : {std::pair<int, int>{1280, 800}, std::pair<int, int>{640, 400}}) {
    const D2CConfig config = makeConfig(size.first, size.second);
    const auto depth = test::makeDepth(size.first, size.second);
    std::vector<uint16_t> filtered(depth.size());
    speckleFilterCpu(depth.data(), filtered.data(), size.first, size.second, {256, 200});
    std::vector<uint16_t> cpu(1280 * 800), gpu(1280 * 800), gpu_filtered(depth.size());
    alignDepthToColorCpu(filtered.data(), cpu.data(), config, buildD2CTables(config));

    CudaDepthStage stage;
    stage.setSpeckle({256, 200});
    std::string error;
    ASSERT_TRUE(stage.setAlignment(config, &error)) << error;
    ASSERT_TRUE(stage.process(depth.data(), size.first, size.second, true, gpu_filtered.data(),
                              gpu.data(), &error))
        << error;
    EXPECT_EQ(filtered, gpu_filtered);
    const auto diff = compareDepth(cpu.data(), gpu.data(), cpu.size());
    EXPECT_EQ(diff.validIoU(), 1.0) << size.first << "x" << size.second;
    EXPECT_EQ(diff.differing, 0u) << size.first << "x" << size.second << " max diff "
                                  << diff.max_abs_diff;
    EXPECT_GT(diff.valid_a, cpu.size() / 2);
  }
}

TEST_F(CudaStages, AlignmentRefusesAnotherDepthSize) {
  CudaDepthStage stage;
  std::string error;
  ASSERT_TRUE(stage.setAlignment(makeConfig(1280, 800), &error)) << error;
  std::vector<uint16_t> depth(640 * 400, 1000), out(1280 * 800);
  EXPECT_FALSE(stage.process(depth.data(), 640, 400, false, nullptr, out.data(), &error));
  EXPECT_NE(error.find("640x400"), std::string::npos) << error;
}

TEST_F(CudaStages, RemapMatchesCpuFromThreeAndFourChannels) {
  const int width = 1280, height = 800;
  const auto lut = buildUndistortLut(test::makeIntrinsic(width, height),
                                     test::makeColorDistortion());
  CudaColorStage stage;
  std::string error;
  ASSERT_TRUE(stage.setLut(lut, &error)) << error;
  for (const int channels : {3, 4}) {
    const size_t pitch = static_cast<size_t>(width) * channels + (channels == 4 ? 256 : 0);
    const auto src = test::makeColor(width, height, channels, pitch);
    std::vector<uint8_t> cpu(static_cast<size_t>(width) * height * 3), gpu(cpu.size());
    remapRgbCpu(src.data(), channels, pitch, cpu.data(), lut);
    ASSERT_TRUE(stage.rectifyHost(src.data(), channels, pitch, gpu.data(), nullptr, &error))
        << error;
    EXPECT_EQ(cpu, gpu) << channels << " channels";
  }
}

TEST_F(CudaStages, DecodedPlanesToRectifiedRgbMatchCpu) {
  const int width = 1280, height = 800;
  const auto lut = buildUndistortLut(test::makeIntrinsic(width, height),
                                     test::makeColorDistortion());
  std::mt19937 rng(5);
  std::uniform_int_distribution<int> sample(0, 255);
  for (const int v_shift : {0, 1}) {
    YcbcrPlanes host;
    host.width = width;
    host.height = height;
    host.h_shift = 1;
    host.v_shift = v_shift;
    std::vector<uint8_t> planes[3];
    const int rows[3] = {height, chromaHeight(host), chromaHeight(host)};
    const size_t pitches[3] = {static_cast<size_t>(width) + 128, 768, 768};
    YcbcrPlanes device = host;
    for (int k = 0; k < 3; ++k) {
      planes[k].resize(pitches[k] * rows[k]);
      for (auto &value : planes[k]) {
        value = static_cast<uint8_t>(sample(rng));
      }
      host.plane[k] = planes[k].data();
      host.pitch[k] = pitches[k];
      uint8_t *dev = nullptr;
      ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dev), planes[k].size()), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(dev, planes[k].data(), planes[k].size(), cudaMemcpyHostToDevice),
                cudaSuccess);
      device.plane[k] = dev;
      device.pitch[k] = pitches[k];
    }
    std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3), cpu(rgb.size()),
        gpu(rgb.size());
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        ycbcrPixelToRgb(host, x, y, &rgb[(static_cast<size_t>(y) * width + x) * 3]);
      }
    }
    remapRgbCpu(rgb.data(), 3, static_cast<size_t>(width) * 3, cpu.data(), lut);
    CudaColorStage stage;
    std::string error;
    ASSERT_TRUE(stage.setLut(lut, &error)) << error;
    ASSERT_TRUE(stage.rectifyYcbcr(device, gpu.data(), nullptr, &error)) << error;
    EXPECT_EQ(cpu, gpu) << "v_shift " << v_shift;
    for (int k = 0; k < 3; ++k) {
      cudaFree(const_cast<uint8_t *>(device.plane[k]));
    }
  }
}

TEST_F(CudaStages, YuvForTheEncoderMatchesCpu) {
  const int width = 1280, height = 800;
  const auto lut = buildUndistortLut(test::makeIntrinsic(width, height),
                                     test::makeColorDistortion());
  const size_t pitch = static_cast<size_t>(width) * 3;
  const auto src = test::makeColor(width, height, 3, pitch);
  std::vector<uint8_t> rgb(pitch * height);
  remapRgbCpu(src.data(), 3, pitch, rgb.data(), lut);

  const size_t y_pitch = width + 64, c_pitch = width / 2 + 32;
  std::vector<uint8_t> y(y_pitch * height, 0), cb(c_pitch * height / 2, 0),
      cr(c_pitch * height / 2, 0);
  for (int by = 0; by < height / 2; ++by) {
    for (int bx = 0; bx < width / 2; ++bx) {
      rgbBlockToYuv420(rgb.data(), pitch, width, height, bx, by, y.data(), y_pitch,
                       &cb[by * c_pitch + bx], &cr[by * c_pitch + bx]);
    }
  }

  Yuv420Planes planes;
  const size_t sizes[3] = {y.size(), cb.size(), cr.size()};
  const size_t pitches[3] = {y_pitch, c_pitch, c_pitch};
  for (int k = 0; k < 3; ++k) {
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&planes.plane[k]), sizes[k]), cudaSuccess);
    ASSERT_EQ(cudaMemset(planes.plane[k], 0, sizes[k]), cudaSuccess);
    planes.pitch[k] = pitches[k];
  }
  CudaColorStage stage;
  std::string error;
  ASSERT_TRUE(stage.setLut(lut, &error)) << error;
  ASSERT_TRUE(stage.rectifyHost(src.data(), 3, pitch, nullptr, &planes, &error)) << error;
  std::vector<uint8_t> gy(y.size()), gcb(cb.size()), gcr(cr.size());
  std::vector<uint8_t> *outs[3] = {&gy, &gcb, &gcr};
  for (int k = 0; k < 3; ++k) {
    ASSERT_EQ(cudaMemcpy(outs[k]->data(), planes.plane[k], sizes[k], cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(planes.plane[k]);
  }
  EXPECT_EQ(y, gy);
  EXPECT_EQ(cb, gcb);
  EXPECT_EQ(cr, gcr);
  // Grey stays grey: JFIF is full range.
  EXPECT_EQ(jfifY(128, 128, 128), 128);
  EXPECT_EQ(jfifCb(128, 128, 128), 128);
  EXPECT_EQ(jfifCr(255, 255, 255), 128);
  EXPECT_EQ(jfifY(255, 255, 255), 255);
}

TEST_F(CudaStages, TimingAtFullResolution) {
  const int width = 1280, height = 800, frames = 30;
  const D2CConfig config = makeConfig(width, height);
  const auto depth = test::makeDepth(width, height);
  std::vector<uint16_t> aligned(depth.size());
  CudaDepthStage depth_stage;
  depth_stage.setSpeckle({256, 200});
  std::string error;
  ASSERT_TRUE(depth_stage.setAlignment(config, &error)) << error;
  ASSERT_TRUE(depth_stage.process(depth.data(), width, height, true, nullptr, aligned.data(),
                                  &error));
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < frames; ++i) {
    ASSERT_TRUE(depth_stage.process(depth.data(), width, height, true, nullptr, aligned.data(),
                                    &error));
  }
  const double depth_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start).count() / frames;

  const auto lut = buildUndistortLut(test::makeIntrinsic(width, height),
                                     test::makeColorDistortion());
  const auto src = test::makeColor(width, height, 3, static_cast<size_t>(width) * 3);
  std::vector<uint8_t> rgb(src.size());
  CudaColorStage color_stage;
  ASSERT_TRUE(color_stage.setLut(lut, &error));
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < frames; ++i) {
    ASSERT_TRUE(color_stage.rectifyHost(src.data(), 3, static_cast<size_t>(width) * 3,
                                        rgb.data(), nullptr, &error));
  }
  const double color_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start).count() / frames;
  std::printf("[ timing ] depth speckle+align %.2f ms/frame, colour upload+remap+download %.2f "
              "ms/frame (wall clock, host copies included)\n", depth_ms, color_ms);
}

}  // namespace
}  // namespace gpu
}  // namespace orbbec_camera
