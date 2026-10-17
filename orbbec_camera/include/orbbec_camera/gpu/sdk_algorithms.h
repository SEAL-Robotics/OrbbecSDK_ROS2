// Copyright 2026 SEAL Robotics
// The ported algorithms are OrbbecSDK v2's: Copyright (c) Orbbec Inc.,
// licensed under the MIT License.
//
// CPU ports of the OrbbecSDK v2.10.6 post-processing the driver used to
// run inside the SDK: colour undistortion (UnDistortionImplGeneric /
// UnDistortionImplSSE), depth-to-colour alignment (AlignImpl::D2C, the SSE
// path the SDK builds on both x86-64 and aarch64) and a speckle filter in
// the role of the SDK's NoiseRemovalFilter. They are the reference the CUDA
// stages are tested against and the fallback where no GPU is present.
//
// The undistortion and alignment ports follow the SDK's arithmetic
// operation for operation, so they reproduce its output bit for bit. The
// SDK's noise filter ships only as a binary (libFilterProcessor's
// softfilterPyramid_lut), so speckleFilterCpu is the textbook speckle filter
// that its two parameters describe; the driver's comparison mode measures
// how far the two differ on real frames.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "libobsensor/h/ObTypes.h"

namespace orbbec_camera {
namespace gpu {

// A 4-connected region of non-zero depth whose neighbouring values differ by
// at most max_diff raw units is a speckle when it has at most max_size
// pixels; every speckle pixel is set to 0.
struct SpeckleParams {
  int max_diff = 256;
  int max_size = 200;
};

void speckleFilterCpu(const uint16_t *in, uint16_t *out, int width, int height,
                      const SpeckleParams &params);

// Everything AlignImpl::initialize() is given for a depth-to-colour (D2C)
// alignment. max_valid_depth is the depth frame's getMaxValidDepthValue():
// raw values at or above it are treated as no measurement.
struct D2CConfig {
  OBCameraIntrinsic depth_intrinsic{};
  OBCameraDistortion depth_distortion{};
  OBCameraIntrinsic color_intrinsic{};
  OBExtrinsic depth_to_color{};
  float depth_unit_mm = 1.0f;
  uint16_t max_valid_depth = 65535;
  bool gap_fill_copy = true;
};

// The SDK's choice of gap filling (Align::createAlignedProfile): splat each
// depth pixel into a 2x2 block when colour has at most twice depth's angular
// resolution, otherwise fill the rectangle between two half-pixel offsets.
bool sdkGapFillCopy(const OBCameraIntrinsic &depth, const OBCameraIntrinsic &color);

// Per-depth-pixel rotation of the (undistorted) ray, AlignImpl's rot_coeff
// tables: X = d * x[k][i] + scaled_trans[0], and likewise for Y and Z.
// One table set for gap-fill copy, two (pixel corners) otherwise.
struct D2CTables {
  int width = 0;
  int height = 0;
  int channels = 1;
  std::vector<float> x[2];
  std::vector<float> y[2];
  std::vector<float> z[2];
  float scaled_trans[3] = {0.0f, 0.0f, 0.0f};
};

D2CTables buildD2CTables(const D2CConfig &config);

// out has color_intrinsic.width x color_intrinsic.height pixels.
void alignDepthToColorCpu(const uint16_t *depth, uint16_t *out, const D2CConfig &config,
                          const D2CTables &tables);

// The SDK's undistortion lookup: for each output pixel the top-left source
// pixel and 5-bit sub-pixel fractions packed as (fy << 5) | fx; oob marks
// output pixels whose source falls outside the image (written as 0).
struct UndistortLut {
  int width = 0;
  int height = 0;
  std::vector<int16_t> xi;
  std::vector<int16_t> yi;
  std::vector<uint16_t> alpha;
  std::vector<uint8_t> oob;
};

UndistortLut buildUndistortLut(const OBCameraIntrinsic &intrinsic,
                               const OBCameraDistortion &distortion);

// Bilinear remap of an 8-bit image with src_channels (3 or 4) channels and
// src_pitch bytes per row into packed 3-channel dst; a fourth source channel
// is dropped.
void remapRgbCpu(const uint8_t *src, int src_channels, size_t src_pitch, uint8_t *dst,
                 const UndistortLut &lut);

}  // namespace gpu
}  // namespace orbbec_camera
