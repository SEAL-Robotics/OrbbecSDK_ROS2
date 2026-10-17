// Copyright 2026 SEAL Robotics
//
// Per-frame differences between the SDK's output and the GPU pipeline's,
// for the driver's comparison mode and the tests.
#pragma once

#include <cstddef>
#include <cstdint>

namespace orbbec_camera {
namespace gpu {

// Depth is valid where it is non-zero.
struct DepthComparison {
  size_t pixels = 0;
  size_t valid_a = 0;
  size_t valid_b = 0;
  size_t valid_both = 0;
  size_t valid_either = 0;
  // Pixels valid in both whose values differ.
  size_t differing = 0;
  int max_abs_diff = 0;
  double mean_abs_diff = 0.0;

  // Intersection over union of the two valid masks; 1 when both are empty.
  double validIoU() const;
};

DepthComparison compareDepth(const uint16_t *a, const uint16_t *b, size_t pixels);

struct ColorComparison {
  size_t samples = 0;
  int max_abs_diff = 0;
  double mean_abs_diff = 0.0;
  // Peak signal-to-noise ratio over all 8-bit samples; infinite when equal.
  double psnr_db = 0.0;
};

ColorComparison compareColor(const uint8_t *a, const uint8_t *b, size_t samples);

}  // namespace gpu
}  // namespace orbbec_camera
