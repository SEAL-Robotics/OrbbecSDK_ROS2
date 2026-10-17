// Copyright 2026 SEAL Robotics
//
// color_math.h against libjpeg itself: the raw YCbCr planes libjpeg
// decodes, converted by ycbcrPixelToRgb, must equal libjpeg's own RGB decode
// of the same file, for each chroma subsampling the camera might send. The
// hardware path feeds NVJPG's planes through the same function.

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "jpeg_test_util.h"
#include "orbbec_camera/gpu/color_math.h"
#include "orbbec_camera/gpu/image_metrics.h"
#include "synthetic_frames.h"

namespace orbbec_camera {
namespace gpu {
namespace {

struct Case {
  int h_samp;
  int v_samp;
  bool fancy;
};

class ColorMath : public ::testing::TestWithParam<Case> {};

TEST_P(ColorMath, RawPlanesConvertExactlyAsLibjpegDecodes) {
  const Case c = GetParam();
  // Odd sizes exercise the last chroma column and row.
  for (const auto &size : {std::pair<int, int>{1280, 800}, std::pair<int, int>{333, 211}}) {
    const auto rgb_in =
        test::makeColor(size.first, size.second, 3, static_cast<size_t>(size.first) * 3);
    const auto jpeg = test::encodeJpeg(rgb_in, size.first, size.second, c.h_samp, c.v_samp, 92);
    const test::RawJpeg raw = test::decodeRaw(jpeg);
    YcbcrPlanes planes;
    for (int k = 0; k < 3; ++k) {
      planes.plane[k] = raw.plane[k].data();
      planes.pitch[k] = raw.pitch[k];
    }
    planes.width = raw.width;
    planes.height = raw.height;
    planes.h_shift = raw.h_shift;
    planes.v_shift = raw.v_shift;
    planes.fancy = c.fancy;

    std::vector<uint8_t> ours(static_cast<size_t>(raw.width) * raw.height * 3);
    for (int y = 0; y < raw.height; ++y) {
      for (int x = 0; x < raw.width; ++x) {
        ycbcrPixelToRgb(planes, x, y, &ours[(static_cast<size_t>(y) * raw.width + x) * 3]);
      }
    }
    // The raw planes come from the default (slow integer) inverse DCT.
    const auto reference = test::decodeRgb(jpeg, c.fancy, JDCT_ISLOW);
    const auto diff = compareColor(reference.data(), ours.data(), ours.size());
    EXPECT_EQ(diff.max_abs_diff, 0) << size.first << "x" << size.second << " h_shift "
                                    << raw.h_shift << " v_shift " << raw.v_shift << " fancy "
                                    << c.fancy << ": PSNR " << diff.psnr_db;
  }
}

INSTANTIATE_TEST_SUITE_P(Subsampling, ColorMath,
                         ::testing::Values(Case{2, 1, false}, Case{2, 2, false},
                                           Case{1, 1, false}, Case{2, 1, true},
                                           Case{2, 2, true}, Case{1, 1, true}));

}  // namespace
}  // namespace gpu
}  // namespace orbbec_camera
