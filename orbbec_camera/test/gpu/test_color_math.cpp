// Copyright 2026 SEAL Robotics
//
// color_math.h against libjpeg itself: the raw YCbCr planes libjpeg
// decodes, converted by ycbcrPixelToRgb, must equal libjpeg's own RGB decode
// of the same file, for each chroma subsampling the camera might send. The
// hardware path feeds NVJPG's planes through the same function.

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <utility>
#include <vector>

#include "jpeg_test_util.h"
#include "orbbec_camera/gpu/color_math.h"
#include "orbbec_camera/gpu/image_metrics.h"
#include "synthetic_frames.h"

namespace orbbec_camera {
namespace gpu {
namespace {

class ColorMath : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(ColorMath, RawPlanesConvertExactlyAsLibjpegDecodes) {
  // Odd sizes exercise the last chroma column and row.
  for (const auto &size : {std::pair<int, int>{1280, 800}, std::pair<int, int>{333, 211}}) {
    const auto rgb_in =
        test::makeColor(size.first, size.second, 3, static_cast<size_t>(size.first) * 3);
    const auto jpeg = test::encodeJpeg(rgb_in, size.first, size.second, GetParam().first,
                                       GetParam().second, 92);
    const test::RawJpeg raw = test::decodeRaw(jpeg);
    YcbcrPlanes planes;
    for (int c = 0; c < 3; ++c) {
      planes.plane[c] = raw.plane[c].data();
      planes.pitch[c] = raw.pitch[c];
    }
    planes.width = raw.width;
    planes.height = raw.height;
    planes.h_shift = raw.h_shift;
    planes.v_shift = raw.v_shift;

    std::vector<uint8_t> ours(static_cast<size_t>(raw.width) * raw.height * 3);
    for (int y = 0; y < raw.height; ++y) {
      for (int x = 0; x < raw.width; ++x) {
        ycbcrPixelToRgb(planes, x, y, &ours[(static_cast<size_t>(y) * raw.width + x) * 3]);
      }
    }
    const cv::Mat encoded(1, static_cast<int>(jpeg.size()), CV_8UC1,
                          const_cast<uint8_t *>(jpeg.data()));
    cv::Mat bgr = cv::imdecode(encoded, cv::IMREAD_COLOR), rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    ASSERT_TRUE(rgb.isContinuous());
    const auto diff = compareColor(rgb.data, ours.data(), ours.size());
    EXPECT_EQ(diff.max_abs_diff, 0) << size.first << "x" << size.second << " h_shift "
                                    << raw.h_shift << " v_shift " << raw.v_shift << ": PSNR "
                                    << diff.psnr_db;
  }
}

INSTANTIATE_TEST_SUITE_P(Subsampling, ColorMath,
                         ::testing::Values(std::pair<int, int>{2, 1}, std::pair<int, int>{2, 2},
                                           std::pair<int, int>{1, 1}));

}  // namespace
}  // namespace gpu
}  // namespace orbbec_camera
