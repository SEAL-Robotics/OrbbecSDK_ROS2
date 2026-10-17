// Copyright 2026 SEAL Robotics
//
// JFIF colour conversions shared by the CUDA kernels and their CPU
// references, in libjpeg's integer arithmetic (jccolor.c, jdcolor.c and
// jdsample.c): a hardware-decoded frame turns into the RGB a libjpeg decode
// of it gives, and the hardware encoder sees the samples a libjpeg encode
// would.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define ORBBEC_HOST_DEVICE __host__ __device__
#else
#define ORBBEC_HOST_DEVICE
#endif

namespace orbbec_camera {
namespace gpu {

ORBBEC_HOST_DEVICE inline int jfifY(int r, int g, int b) {
  return (19595 * r + 38470 * g + 7471 * b + (1 << 15)) >> 16;
}

ORBBEC_HOST_DEVICE inline int jfifCb(int r, int g, int b) {
  return (-11059 * r - 21709 * g + 32768 * b + (128 << 16) + (1 << 15) - 1) >> 16;
}

ORBBEC_HOST_DEVICE inline int jfifCr(int r, int g, int b) {
  return (32768 * r - 27439 * g - 5329 * b + (128 << 16) + (1 << 15) - 1) >> 16;
}

// One 2x2 block of packed RGB at (bx, by) in block units: four luma samples
// and the rounded mean of its four chroma samples. Blocks on an odd right or
// bottom edge repeat the last column or row.
ORBBEC_HOST_DEVICE inline void rgbBlockToYuv420(const uint8_t *rgb, size_t rgb_pitch, int width,
                                                int height, int bx, int by, uint8_t *y_plane,
                                                size_t y_pitch, uint8_t *cb, uint8_t *cr) {
  int cb_sum = 0;
  int cr_sum = 0;
  for (int dy = 0; dy < 2; ++dy) {
    const int py = (2 * by + dy < height) ? 2 * by + dy : height - 1;
    for (int dx = 0; dx < 2; ++dx) {
      const int px = (2 * bx + dx < width) ? 2 * bx + dx : width - 1;
      const uint8_t *p = rgb + py * rgb_pitch + px * 3;
      const int r = p[0];
      const int g = p[1];
      const int b = p[2];
      if (2 * by + dy < height && 2 * bx + dx < width) {
        y_plane[py * y_pitch + px] = static_cast<uint8_t>(jfifY(r, g, b));
      }
      cb_sum += jfifCb(r, g, b);
      cr_sum += jfifCr(r, g, b);
    }
  }
  *cb = static_cast<uint8_t>((cb_sum + 2) >> 2);
  *cr = static_cast<uint8_t>((cr_sum + 2) >> 2);
}

// Planes of a decoded JPEG: full-resolution Y, and Cb and Cr subsampled 2:1
// horizontally (h_shift 1) and, for 4:2:0, vertically (v_shift 1).
struct YcbcrPlanes {
  const uint8_t *plane[3] = {nullptr, nullptr, nullptr};
  size_t pitch[3] = {0, 0, 0};
  int width = 0;
  int height = 0;
  int h_shift = 1;
  int v_shift = 0;
};

ORBBEC_HOST_DEVICE inline int chromaWidth(const YcbcrPlanes &p) {
  return (p.width + (1 << p.h_shift) - 1) >> p.h_shift;
}

ORBBEC_HOST_DEVICE inline int chromaHeight(const YcbcrPlanes &p) {
  return (p.height + (1 << p.v_shift) - 1) >> p.v_shift;
}

// libjpeg's "fancy" (triangle-filter) upsampling of one chroma sample at
// full-resolution (x, y): h2v1_fancy_upsample and h2v2_fancy_upsample, with
// the image's first and last rows repeated as their context.
ORBBEC_HOST_DEVICE inline int fancyChroma(const uint8_t *plane, size_t pitch, const YcbcrPlanes &p,
                                          int x, int y) {
  const int cw = chromaWidth(p);
  const int j = x >> p.h_shift;
  if (p.h_shift == 0 && p.v_shift == 0) {
    return plane[y * pitch + x];
  }
  if (p.v_shift == 0) {
    const uint8_t *row = plane + y * pitch;
    const int c = row[j];
    if ((x & 1) == 0) {
      return j == 0 ? c : (3 * c + row[j - 1] + 1) >> 2;
    }
    return j == cw - 1 ? c : (3 * c + row[j + 1] + 2) >> 2;
  }
  const int ch = chromaHeight(p);
  const int i = y >> 1;
  int i2 = (y & 1) == 0 ? i - 1 : i + 1;
  i2 = i2 < 0 ? 0 : (i2 > ch - 1 ? ch - 1 : i2);
  const uint8_t *near_row = plane + i * pitch;
  const uint8_t *far_row = plane + i2 * pitch;
  const int this_sum = near_row[j] * 3 + far_row[j];
  if ((x & 1) == 0) {
    if (j == 0) {
      return (this_sum * 4 + 8) >> 4;
    }
    return (this_sum * 3 + near_row[j - 1] * 3 + far_row[j - 1] + 8) >> 4;
  }
  if (j == cw - 1) {
    return (this_sum * 4 + 7) >> 4;
  }
  return (this_sum * 3 + near_row[j + 1] * 3 + far_row[j + 1] + 7) >> 4;
}

ORBBEC_HOST_DEVICE inline uint8_t clampByte(int v) {
  return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// jdcolor.c ycc_rgb_convert, from its tables' definitions.
ORBBEC_HOST_DEVICE inline void jfifToRgb(int y, int cb, int cr, uint8_t *rgb) {
  const int x_cb = cb - 128;
  const int x_cr = cr - 128;
  const int cr_r = (91881 * x_cr + (1 << 15)) >> 16;
  const int cb_b = (116130 * x_cb + (1 << 15)) >> 16;
  const int cb_g = -22554 * x_cb + (1 << 15);
  const int cr_g = -46802 * x_cr;
  rgb[0] = clampByte(y + cr_r);
  rgb[1] = clampByte(y + ((cb_g + cr_g) >> 16));
  rgb[2] = clampByte(y + cb_b);
}

ORBBEC_HOST_DEVICE inline void ycbcrPixelToRgb(const YcbcrPlanes &p, int x, int y, uint8_t *rgb) {
  const int luma = p.plane[0][y * p.pitch[0] + x];
  const int cb = fancyChroma(p.plane[1], p.pitch[1], p, x, y);
  const int cr = fancyChroma(p.plane[2], p.pitch[2], p, x, y);
  jfifToRgb(luma, cb, cr, rgb);
}

}  // namespace gpu
}  // namespace orbbec_camera
