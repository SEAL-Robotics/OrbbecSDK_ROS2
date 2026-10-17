// Copyright 2026 SEAL Robotics
// The alignment and remap below follow OrbbecSDK v2.10.6 AlignImpl.cpp and
// UnDistortionImplGeneric.cpp: Copyright (c) Orbbec Inc., licensed under the
// MIT License.
//
// Built with -ffp-contract=off: whether the per-pixel alignment fuses a
// multiply and an add must follow the SDK, not this compiler's choice.

#include <algorithm>
#include <cmath>
#include <limits>

#include "orbbec_camera/gpu/sdk_algorithms.h"

namespace orbbec_camera {
namespace gpu {
namespace {

// AlignImpl.hpp EPSILON, a double: the SDK compares float depths against it.
constexpr double kAlignEpsilon = 1e-6;
constexpr int kInterBits = 5;
constexpr int kInterTab = 1 << kInterBits;

// static_cast<int> of a float the way both SDK targets behave for the
// values that matter: out of int range never lands inside the image.
bool truncToInt(float value, int &out) {
  if (!(value > -2147483648.0f && value < 2147483648.0f)) {
    return false;
  }
  out = static_cast<int>(value);
  return true;
}

bool truncToInt(double value, int &out) {
  if (!(value > -2147483648.0 && value < 2147483648.0)) {
    return false;
  }
  out = static_cast<int>(value);
  return true;
}

// static_cast<uint16_t>(float) as compiled for the SDK: a 32-bit conversion
// truncated to 16 bits.
uint16_t depthToU16(float z) {
  if (!(z < 4294967296.0f)) {
    return 0xFFFF;
  }
  return static_cast<uint16_t>(static_cast<uint32_t>(z));
}

struct Projection {
  float x;
  float y;
  float z;
};

// a * b + c as the SDK's build computes it: on aarch64 GCC fuses its
// SSE2NEON multiply and add into one FMA, x86-64 SSE keeps them separate
// (ORBBEC_SDK_FUSED_MULTIPLY_ADD, set by CMakeLists.txt; both pinned by
// test_sdk_algorithms against the SDK binary).
inline float mulAdd(float a, float b, float c) {
#if defined(ORBBEC_SDK_FUSED_MULTIPLY_ADD)
  return std::fma(a, b, c);
#else
  return a * b + c;
#endif
}

// AlignImpl::CalcNormCorrdWithSSE() and the pixel projection after it.
Projection project(float depth, float cx_coeff, float cy_coeff, float cz_coeff,
                   const float scaled_trans[3], const OBCameraIntrinsic &color) {
  const float y_cam = mulAdd(depth, cy_coeff, scaled_trans[1]);
  const float x_cam = mulAdd(depth, cx_coeff, scaled_trans[0]);
  const float z_cam = mulAdd(depth, cz_coeff, scaled_trans[2]);
  const float nx = x_cam / z_cam;
  const float ny = y_cam / z_cam;
  return {mulAdd(nx, color.fx, color.cx), mulAdd(ny, color.fy, color.cy), z_cam};
}

}  // namespace

void speckleFilterCpu(const uint16_t *in, uint16_t *out, int width, int height,
                      const SpeckleParams &params) {
  const size_t n = static_cast<size_t>(width) * height;
  std::copy(in, in + n, out);
  std::vector<uint8_t> visited(n, 0);
  std::vector<int> region;
  std::vector<int> stack;
  for (size_t start = 0; start < n; ++start) {
    if (visited[start] || in[start] == 0) {
      continue;
    }
    region.clear();
    stack.clear();
    stack.push_back(static_cast<int>(start));
    visited[start] = 1;
    while (!stack.empty()) {
      const int p = stack.back();
      stack.pop_back();
      region.push_back(p);
      const int px = p % width;
      const int py = p / width;
      const int neighbours[4][2] = {{px - 1, py}, {px + 1, py}, {px, py - 1}, {px, py + 1}};
      for (const auto &q : neighbours) {
        if (q[0] < 0 || q[0] >= width || q[1] < 0 || q[1] >= height) {
          continue;
        }
        const int qi = q[1] * width + q[0];
        if (visited[qi] || in[qi] == 0) {
          continue;
        }
        if (std::abs(static_cast<int>(in[p]) - static_cast<int>(in[qi])) > params.max_diff) {
          continue;
        }
        visited[qi] = 1;
        stack.push_back(qi);
      }
    }
    if (static_cast<int>(region.size()) <= params.max_size) {
      for (const int p : region) {
        out[p] = 0;
      }
    }
  }
}

// AlignImpl::D2C() with withSSE (the default): LinearD2CWithSSE and its
// FillSingleChannelWithSSE / FillMultiChannelWithSSE.
void alignDepthToColorCpu(const uint16_t *depth, uint16_t *out, const D2CConfig &config,
                          const D2CTables &tables) {
  const OBCameraIntrinsic &color = config.color_intrinsic;
  const int width = color.width;
  const int height = color.height;
  const size_t pixnum = static_cast<size_t>(width) * height;
  std::fill(out, out + pixnum, static_cast<uint16_t>(0xFFFF));

  const size_t depth_pixels = static_cast<size_t>(tables.width) * tables.height;
  for (size_t i = 0; i < depth_pixels; ++i) {
    uint16_t raw = depth[i];
    if (raw >= config.max_valid_depth) {
      raw = 0;
    }
    const float d = static_cast<float>(raw);

    if (tables.channels == 1) {
      const Projection p =
          project(d, tables.x[0][i], tables.y[0][i], tables.z[0][i], tables.scaled_trans, color);
      if (p.z < kAlignEpsilon) {
        continue;
      }
      int u_rgb, v_rgb;
      if (!truncToInt(p.x + 0.5f, u_rgb) || !truncToInt(p.y + 0.5f, v_rgb)) {
        continue;
      }
      if (u_rgb < 0 || u_rgb >= width || v_rgb < 0 || v_rgb >= height) {
        continue;
      }
      const uint16_t cur_depth = depthToU16(p.z);
      const size_t pos = static_cast<size_t>(v_rgb) * width + u_rgb;
      out[pos] = std::min(out[pos], cur_depth);
      if ((u_rgb + 1) < width) {
        out[pos + 1] = std::min(out[pos + 1], cur_depth);
        if ((v_rgb + 1) < height) {
          out[pos + width + 1] = std::min(out[pos + width + 1], cur_depth);
        }
      }
      if ((v_rgb + 1) < height) {
        out[pos + width] = std::min(out[pos + width], cur_depth);
      }
      continue;
    }

    bool valid = true;
    int u_rgb[2] = {-1, -1};
    int v_rgb[2] = {-1, -1};
    uint16_t cur_depth = 65535;
    for (int chl = 0; chl < 2; chl++) {
      const Projection p = project(d, tables.x[chl][i], tables.y[chl][i], tables.z[chl][i],
                                   tables.scaled_trans, color);
      if (p.z < kAlignEpsilon) {
        valid = false;
        break;
      }
      // The multi-channel fill adds a double 0.5, unlike the single one.
      if (!truncToInt(static_cast<double>(p.x) + 0.5, u_rgb[chl]) ||
          !truncToInt(static_cast<double>(p.y) + 0.5, v_rgb[chl])) {
        valid = false;
        break;
      }
      cur_depth = std::min(cur_depth, depthToU16(p.z));
    }
    if (!valid) {
      continue;
    }
    const int u0 = std::max(0, std::min(u_rgb[0], u_rgb[1]));
    const int v0 = std::max(0, std::min(v_rgb[0], v_rgb[1]));
    const int u1 = std::min(std::max(u_rgb[0], u_rgb[1]), width - 1);
    const int v1 = std::min(std::max(v_rgb[0], v_rgb[1]), height - 1);
    for (int vr = v0; vr <= v1; vr++) {
      for (int ur = u0; ur <= u1; ur++) {
        const size_t pos = static_cast<size_t>(vr) * width + ur;
        out[pos] = std::min(out[pos], cur_depth);
      }
    }
  }

  for (size_t idx = 0; idx < pixnum; idx++) {
    if (config.max_valid_depth <= out[idx]) {
      out[idx] = 0;
    }
  }
}

// UnDistortionImplGeneric::remapNChannelBilinear() then the zero fill of
// out-of-bounds pixels in undistort().
void remapRgbCpu(const uint8_t *src, int src_channels, size_t src_pitch, uint8_t *dst,
                 const UndistortLut &lut) {
  const int src_w = lut.width;
  const int src_h = lut.height;
  for (int v = 0; v < lut.height; v++) {
    for (int u = 0; u < lut.width; u++) {
      const size_t idx = static_cast<size_t>(v) * lut.width + u;
      uint8_t *o = dst + idx * 3;
      if (lut.oob[idx]) {
        o[0] = o[1] = o[2] = 0;
        continue;
      }
      const int x0 = lut.xi[idx];
      const int y0 = lut.yi[idx];
      const uint16_t alpha = lut.alpha[idx];
      const int fx = alpha & (kInterTab - 1);
      const int fy = alpha >> kInterBits;
      const int x1 = std::min(x0 + 1, src_w - 1);
      const int y1 = std::min(y0 + 1, src_h - 1);
      const int fxi = kInterTab - fx;
      const int fyi = kInterTab - fy;
      const int w00 = fxi * fyi;
      const int w10 = fx * fyi;
      const int w01 = fxi * fy;
      const int w11 = fx * fy;
      const uint8_t *row0 = src + static_cast<size_t>(y0) * src_pitch;
      const uint8_t *row1 = src + static_cast<size_t>(y1) * src_pitch;
      const uint8_t *p00 = row0 + x0 * src_channels;
      const uint8_t *p10 = row0 + x1 * src_channels;
      const uint8_t *p01 = row1 + x0 * src_channels;
      const uint8_t *p11 = row1 + x1 * src_channels;
      for (int c = 0; c < 3; c++) {
        o[c] = static_cast<uint8_t>(
            (w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c] + 512) >> 10);
      }
    }
  }
}

}  // namespace gpu
}  // namespace orbbec_camera
