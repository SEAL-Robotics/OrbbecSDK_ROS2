// Copyright 2026 SEAL Robotics
// Ported from OrbbecSDK v2.10.6 src/filter/publicfilters (AlignImpl.cpp,
// Align.cpp, UnDistortionImplGeneric.cpp): Copyright (c) Orbbec Inc.,
// licensed under the MIT License.
//
// Lookup tables the SDK builds once per camera configuration. They are
// scalar code in the SDK, compiled with the toolchain's default
// floating-point contraction, and so are these: the same expressions, built
// the same way, give the same tables on the same architecture.

#include <algorithm>
#include <cmath>
#include <cstring>

#include "orbbec_camera/gpu/sdk_algorithms.h"

namespace orbbec_camera {
namespace gpu {
namespace {

// UnDistortionImplGeneric::INTER_BITS
constexpr int kInterBits = 5;
constexpr int kInterTab = 1 << kInterBits;

// AlignImpl.cpp addDistortion()
void alignAddDistortion(const OBCameraDistortion &distort_param, const float pt_ud[2],
                        float pt_d[2]) {
  float k1 = distort_param.k1, k2 = distort_param.k2, k3 = distort_param.k3;
  float k4 = distort_param.k4, k5 = distort_param.k5, k6 = distort_param.k6;
  float p1 = distort_param.p1, p2 = distort_param.p2;

  const float r2 = pt_ud[0] * pt_ud[0] + pt_ud[1] * pt_ud[1];

  if ((distort_param.model == OB_DISTORTION_BROWN_CONRADY) ||
      (distort_param.model == OB_DISTORTION_BROWN_CONRADY_K6)) {
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;

    float k_diff = 1 + k1 * r2 + k2 * r4 + k3 * r6;
    if (distort_param.model == OB_DISTORTION_BROWN_CONRADY_K6) {
      k_diff /= (1 + k4 * r2 + k5 * r4 + k6 * r6);
    }

    const float t_x = p2 * (r2 + 2 * pt_ud[0] * pt_ud[0]) + 2 * p1 * pt_ud[0] * pt_ud[1];
    const float t_y = p1 * (r2 + 2 * pt_ud[1] * pt_ud[1]) + 2 * p2 * pt_ud[0] * pt_ud[1];

    pt_d[0] = pt_ud[0] * k_diff + t_x;
    pt_d[1] = pt_ud[1] * k_diff + t_y;
  } else if (distort_param.model == OB_DISTORTION_KANNALA_BRANDT4) {
    const double r = sqrt(r2);
    const double theta = atan(r);
    const double theta2 = theta * theta;
    const double theta4 = theta2 * theta2;
    const double theta6 = theta2 * theta4;
    const double theta8 = theta4 * theta4;

    const double k_diff =
        theta * (1 + (k1 * theta2) + (k2 * theta4) + (k3 * theta6) + (k4 * theta8));

    pt_d[0] = static_cast<float>(k_diff / r * pt_ud[0]);
    pt_d[1] = static_cast<float>(k_diff / r * pt_ud[1]);
  }
}

// AlignImpl.cpp removeDistortion()
void alignRemoveDistortion(const OBCameraDistortion &distort_param, const float pt_d[2],
                           float pt_ud[2]) {
  const float epsilon = 1e-6f;
  const int max_iteration = 20;
  float tmp_p_ud[2] = {pt_d[0], pt_d[1]};
  std::memset(pt_ud, 0, sizeof(float) * 2);
  alignAddDistortion(distort_param, tmp_p_ud, pt_ud);

  pt_ud[0] = pt_ud[0] - tmp_p_ud[0];
  pt_ud[1] = pt_ud[1] - tmp_p_ud[1];
  int it = 0;
  float err = std::fabs(tmp_p_ud[0] + pt_ud[0] - pt_d[0]) +
              std::fabs(tmp_p_ud[1] + pt_ud[1] - pt_d[1]);
  while (err > epsilon && it++ < max_iteration) {
    tmp_p_ud[0] = pt_d[0] - pt_ud[0];
    tmp_p_ud[1] = pt_d[1] - pt_ud[1];
    alignAddDistortion(distort_param, tmp_p_ud, pt_ud);

    pt_ud[0] = pt_ud[0] - tmp_p_ud[0];
    pt_ud[1] = pt_ud[1] - tmp_p_ud[1];
    err = std::fabs(tmp_p_ud[0] + pt_ud[0] - pt_d[0]) +
          std::fabs(tmp_p_ud[1] + pt_ud[1] - pt_d[1]);
  }

  pt_ud[0] = tmp_p_ud[0];
  pt_ud[1] = tmp_p_ud[1];
}

// UnDistortionImplGeneric.cpp applyDistortion()
void undistortApplyDistortion(const OBCameraDistortion &d, float xn, float yn, float &xd,
                              float &yd) {
  float r2 = xn * xn + yn * yn;

  switch (d.model) {
    case OB_DISTORTION_NONE:
      xd = xn;
      yd = yn;
      return;

    case OB_DISTORTION_BROWN_CONRADY:
    case OB_DISTORTION_MODIFIED_BROWN_CONRADY:
    case OB_DISTORTION_INVERSE_BROWN_CONRADY: {
      float r4 = r2 * r2;
      float r6 = r4 * r2;
      float radial = 1.0f + d.k1 * r2 + d.k2 * r4 + d.k3 * r6;
      xd = xn * radial + 2.0f * d.p1 * xn * yn + d.p2 * (r2 + 2.0f * xn * xn);
      yd = yn * radial + d.p1 * (r2 + 2.0f * yn * yn) + 2.0f * d.p2 * xn * yn;
      return;
    }

    case OB_DISTORTION_BROWN_CONRADY_K6: {
      float r4 = r2 * r2;
      float r6 = r4 * r2;
      float num = 1.0f + d.k1 * r2 + d.k2 * r4 + d.k3 * r6;
      float den = 1.0f + d.k4 * r2 + d.k5 * r4 + d.k6 * r6;
      float radial = (std::abs(den) > 1e-8f) ? (num / den) : num;
      xd = xn * radial + 2.0f * d.p1 * xn * yn + d.p2 * (r2 + 2.0f * xn * xn);
      yd = yn * radial + d.p1 * (r2 + 2.0f * yn * yn) + 2.0f * d.p2 * xn * yn;
      return;
    }

    case OB_DISTORTION_KANNALA_BRANDT4: {
      float r = std::sqrt(r2);
      float theta = std::atan2(r, 1.0f);
      float t2 = theta * theta;
      float t4 = t2 * t2;
      float t6 = t4 * t2;
      float t8 = t4 * t4;
      float theta_d = theta * (1.0f + d.k1 * t2 + d.k2 * t4 + d.k3 * t6 + d.k4 * t8);
      float scale = (r > 1e-6f) ? (theta_d / r) : 1.0f;
      xd = xn * scale;
      yd = yn * scale;
      return;
    }

    default:
      xd = xn;
      yd = yn;
      return;
  }
}

// Align.cpp estimateFOV()
void estimateFov(const OBCameraIntrinsic &intrin, float *fovs) {
  fovs[0] = atan2f(intrin.cx, intrin.fx) + atan2f(intrin.width - intrin.cx - 1.f, intrin.fx);
  fovs[1] = atan2f(intrin.cy, intrin.fy) + atan2f(intrin.height - intrin.cy - 1.f, intrin.fy);
}

}  // namespace

bool sdkGapFillCopy(const OBCameraIntrinsic &depth, const OBCameraIntrinsic &color) {
  float ori_fov[2], to_fov[2];
  estimateFov(depth, ori_fov);
  estimateFov(color, to_fov);
  const float ori_pix_per_ang[2] = {depth.width / ori_fov[0], depth.height / ori_fov[1]};
  const float to_pix_per_ang[2] = {color.width / to_fov[0], color.height / to_fov[1]};
  return (ori_pix_per_ang[0] / to_pix_per_ang[0] > 0.5f) &&
         (ori_pix_per_ang[1] / to_pix_per_ang[1] > 0.5f);
}

// AlignImpl::initialize() and prepareDepthResolution()
D2CTables buildD2CTables(const D2CConfig &config) {
  const OBCameraIntrinsic &intr = config.depth_intrinsic;
  const OBCameraDistortion &disto = config.depth_distortion;
  const OBExtrinsic &transform = config.depth_to_color;
  const bool need_to_undistort_depth = (disto.k1 != 0) || (disto.k2 != 0) || (disto.k3 != 0) ||
                                       (disto.k4 != 0) || (disto.k5 != 0) || (disto.k6 != 0) ||
                                       (disto.p1 != 0) || (disto.p2 != 0);

  D2CTables tables;
  tables.width = intr.width;
  tables.height = intr.height;
  tables.channels = config.gap_fill_copy ? 1 : 2;
  for (int i = 0; i < 3; ++i) {
    tables.scaled_trans[i] = transform.trans[i] / config.depth_unit_mm;
  }

  const size_t stride = static_cast<size_t>(intr.width);
  const size_t coeff_num = static_cast<size_t>(intr.height) * stride;
  for (int i = 0; i < tables.channels; i++) {
    tables.x[i].resize(coeff_num);
    tables.y[i].resize(coeff_num);
    tables.z[i].resize(coeff_num);
    const int mutliplier = (config.gap_fill_copy ? 0 : (i ? 1 : -1));
    for (int v = 0; v < intr.height; ++v) {
      float *dst1 = tables.x[i].data() + v * stride;
      float *dst2 = tables.y[i].data() + v * stride;
      float *dst3 = tables.z[i].data() + v * stride;
      float y = (v + mutliplier * 0.5f - intr.cy) / intr.fy;
      for (int u = 0; u < intr.width; ++u) {
        float x = (u + mutliplier * 0.5f - intr.cx) / intr.fx;

        float x_ud = x, y_ud = y;
        if (need_to_undistort_depth) {
          float pt_d[2] = {x, y};
          float pt_ud[2];
          alignRemoveDistortion(disto, pt_d, pt_ud);
          x_ud = pt_ud[0];
          y_ud = pt_ud[1];
        }

        *dst1 = transform.rot[0] * x_ud + transform.rot[1] * y_ud + transform.rot[2];
        *dst2 = transform.rot[3] * x_ud + transform.rot[4] * y_ud + transform.rot[5];
        *dst3 = transform.rot[6] * x_ud + transform.rot[7] * y_ud + transform.rot[8];
        dst1++;
        dst2++;
        dst3++;
      }
    }
  }
  return tables;
}

// UnDistortionImplGeneric::buildLUT() without a new camera matrix, the only
// way the driver configures the filter for this camera.
UndistortLut buildUndistortLut(const OBCameraIntrinsic &intrin,
                               const OBCameraDistortion &disto) {
  UndistortLut lut;
  const int src_w = intrin.width;
  const int src_h = intrin.height;
  lut.width = src_w;
  lut.height = src_h;
  const size_t n = static_cast<size_t>(src_w) * src_h;
  lut.xi.assign(n, 0);
  lut.yi.assign(n, 0);
  lut.alpha.assign(n, 0);
  lut.oob.assign(n, 0);

  const float fx = intrin.fx, fy = intrin.fy, cx = intrin.cx, cy = intrin.cy;
  const float vfx = fx, vfy = fy, vcx = cx, vcy = cy;

  for (int v = 0; v < src_h; v++) {
    for (int u = 0; u < src_w; u++) {
      float xn = (static_cast<float>(u) - vcx) / vfx;
      float yn = (static_cast<float>(v) - vcy) / vfy;
      float xd, yd;
      undistortApplyDistortion(disto, xn, yn, xd, yd);

      float xs = xd * fx + cx;
      float ys = yd * fy + cy;

      const size_t idx = static_cast<size_t>(v) * src_w + u;

      if (xs < 0.0f || xs > static_cast<float>(src_w - 1) || ys < 0.0f ||
          ys > static_cast<float>(src_h - 1)) {
        lut.oob[idx] = 1;
        continue;
      }

      xs = std::max(0.0f, std::min(xs, static_cast<float>(src_w - 1)));
      ys = std::max(0.0f, std::min(ys, static_cast<float>(src_h - 1)));

      int x0 = static_cast<int>(xs);
      int y0 = static_cast<int>(ys);
      int fx5 = static_cast<int>((xs - x0) * kInterTab + 0.5f);
      int fy5 = static_cast<int>((ys - y0) * kInterTab + 0.5f);
      if (fx5 >= kInterTab) {
        fx5 = 0;
        x0 = std::min(x0 + 1, src_w - 1);
      }
      if (fy5 >= kInterTab) {
        fy5 = 0;
        y0 = std::min(y0 + 1, src_h - 1);
      }
      x0 = std::min(x0, src_w - 1);
      y0 = std::min(y0, src_h - 1);

      lut.xi[idx] = static_cast<int16_t>(x0);
      lut.yi[idx] = static_cast<int16_t>(y0);
      lut.alpha[idx] = static_cast<uint16_t>((fy5 << kInterBits) | fx5);
    }
  }
  return lut;
}

}  // namespace gpu
}  // namespace orbbec_camera
