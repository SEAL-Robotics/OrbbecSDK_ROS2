// Copyright 2026 SEAL Robotics
//
// Deterministic frames and camera models for the GPU pipeline tests, close
// to the Gemini 338Le at 1280x800: a scene of slanted planes with depth
// edges, holes, isolated blobs and sensor-like noise, and a textured colour
// image with sharp edges for the remap to interpolate.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "libobsensor/h/ObTypes.h"

namespace orbbec_camera {
namespace gpu {
namespace test {

inline OBCameraIntrinsic makeIntrinsic(int width, int height, float scale = 1.0f) {
  OBCameraIntrinsic k{};
  k.fx = 640.3f * scale * width / 1280.0f;
  k.fy = 640.9f * scale * height / 800.0f;
  k.cx = 639.4f * width / 1280.0f;
  k.cy = 401.7f * height / 800.0f;
  k.width = static_cast<int16_t>(width);
  k.height = static_cast<int16_t>(height);
  return k;
}

inline OBCameraDistortion makeColorDistortion() {
  OBCameraDistortion d{};
  d.k1 = -0.0563f;
  d.k2 = 0.0661f;
  d.k3 = -0.0214f;
  d.p1 = 0.00031f;
  d.p2 = -0.00047f;
  d.model = OB_DISTORTION_BROWN_CONRADY;
  return d;
}

// A small rotation and the 338Le's roughly 32 mm depth-to-colour baseline.
inline OBExtrinsic makeDepthToColor() {
  const double ax = 0.0021, ay = -0.0035, az = 0.0012;
  const double cx = std::cos(ax), sx = std::sin(ax);
  const double cy = std::cos(ay), sy = std::sin(ay);
  const double cz = std::cos(az), sz = std::sin(az);
  const double r[9] = {cy * cz,
                       -cy * sz,
                       sy,
                       sx * sy * cz + cx * sz,
                       -sx * sy * sz + cx * cz,
                       -sx * cy,
                       -cx * sy * cz + sx * sz,
                       cx * sy * sz + sx * cz,
                       cx * cy};
  OBExtrinsic e{};
  for (int i = 0; i < 9; ++i) {
    e.rot[i] = static_cast<float>(r[i]);
  }
  e.trans[0] = -32.07f;
  e.trans[1] = 0.41f;
  e.trans[2] = -0.23f;
  return e;
}

inline std::vector<uint16_t> makeDepth(int width, int height, uint32_t seed = 7) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> noise(0.0f, 2.5f);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  std::vector<uint16_t> depth(static_cast<size_t>(width) * height, 0);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const float x = static_cast<float>(u) / width;
      const float y = static_cast<float>(v) / height;
      float z = 900.0f + 350.0f * x + 120.0f * y;
      if (x > 0.3f && x < 0.55f && y > 0.25f && y < 0.7f) {
        z = 520.0f + 40.0f * y;
      }
      if (x > 0.7f && y > 0.6f) {
        z = 2300.0f - 400.0f * x;
      }
      if ((x - 0.15f) * (x - 0.15f) + (y - 0.8f) * (y - 0.8f) < 0.004f) {
        z = 0.0f;
      }
      if (unit(rng) < 0.01f) {
        z = 0.0f;
      }
      depth[static_cast<size_t>(v) * width + u] =
          z <= 0.0f ? 0 : static_cast<uint16_t>(std::max(1.0f, z + noise(rng)));
    }
  }
  // Floating blobs of assorted sizes, far from their surroundings.
  std::uniform_int_distribution<int> px(2, width - 40);
  std::uniform_int_distribution<int> py(2, height - 40);
  std::uniform_int_distribution<int> side(1, 22);
  for (int b = 0; b < 160; ++b) {
    const int x0 = px(rng), y0 = py(rng), w = side(rng), h = side(rng);
    const uint16_t value = static_cast<uint16_t>(300 + (b * 37) % 2500);
    for (int v = y0; v < y0 + h; ++v) {
      for (int u = x0; u < x0 + w; ++u) {
        depth[static_cast<size_t>(v) * width + u] = value;
      }
    }
  }
  return depth;
}

// channels 3 or 4; pitch >= width * channels.
inline std::vector<uint8_t> makeColor(int width, int height, int channels, size_t pitch,
                                      uint32_t seed = 11) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> grain(-12, 12);
  std::vector<uint8_t> image(pitch * height, 0);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      uint8_t *p = image.data() + v * pitch + u * channels;
      const bool checker = ((u / 37) + (v / 29)) % 2 == 0;
      const int base[3] = {(u * 255) / width, (v * 255) / height, checker ? 220 : 35};
      for (int c = 0; c < 3; ++c) {
        p[c] = static_cast<uint8_t>(std::min(255, std::max(0, base[c] + grain(rng))));
      }
      if (channels == 4) {
        p[3] = 255;
      }
    }
  }
  return image;
}

}  // namespace test
}  // namespace gpu
}  // namespace orbbec_camera
