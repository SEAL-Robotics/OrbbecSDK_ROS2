// Copyright 2026 SEAL Robotics

#include "orbbec_camera/gpu/image_metrics.h"

#include <cmath>
#include <cstdlib>
#include <limits>

namespace orbbec_camera {
namespace gpu {

double DepthComparison::validIoU() const {
  if (valid_either == 0) {
    return 1.0;
  }
  return static_cast<double>(valid_both) / static_cast<double>(valid_either);
}

DepthComparison compareDepth(const uint16_t *a, const uint16_t *b, size_t pixels) {
  DepthComparison result;
  result.pixels = pixels;
  uint64_t abs_sum = 0;
  for (size_t i = 0; i < pixels; ++i) {
    const bool va = a[i] != 0;
    const bool vb = b[i] != 0;
    result.valid_a += va;
    result.valid_b += vb;
    result.valid_either += (va || vb);
    if (!(va && vb)) {
      continue;
    }
    ++result.valid_both;
    const int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    if (diff != 0) {
      ++result.differing;
      abs_sum += static_cast<uint64_t>(diff);
      if (diff > result.max_abs_diff) {
        result.max_abs_diff = diff;
      }
    }
  }
  if (result.valid_both > 0) {
    result.mean_abs_diff =
        static_cast<double>(abs_sum) / static_cast<double>(result.valid_both);
  }
  return result;
}

ColorComparison compareColor(const uint8_t *a, const uint8_t *b, size_t samples) {
  ColorComparison result;
  result.samples = samples;
  uint64_t abs_sum = 0;
  uint64_t sq_sum = 0;
  for (size_t i = 0; i < samples; ++i) {
    const int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    abs_sum += static_cast<uint64_t>(diff);
    sq_sum += static_cast<uint64_t>(diff) * static_cast<uint64_t>(diff);
    if (diff > result.max_abs_diff) {
      result.max_abs_diff = diff;
    }
  }
  if (samples == 0) {
    return result;
  }
  result.mean_abs_diff = static_cast<double>(abs_sum) / static_cast<double>(samples);
  const double mse = static_cast<double>(sq_sum) / static_cast<double>(samples);
  result.psnr_db = mse == 0.0 ? std::numeric_limits<double>::infinity()
                              : 10.0 * std::log10(255.0 * 255.0 / mse);
  return result;
}

}  // namespace gpu
}  // namespace orbbec_camera
