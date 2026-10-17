// Copyright 2026 SEAL Robotics
//
// The two halves of the driver's GPU pipeline, each picking the best
// backend the build and the machine offer:
//
//   DepthPipeline  speckle filter + depth-to-colour alignment on CUDA, or
//                  the CPU references where there is no GPU.
//   ColorPipeline  MJPG decode, undistortion and JPEG encode of the
//                  rectified image: NVJPG decode and encode with VIC colour
//                  conversion and a CUDA remap on a Jetson; a CPU libjpeg
//                  codec around the CUDA remap elsewhere; all CPU without a
//                  GPU.
//
// Objects are not thread-safe; the driver keeps each on the one thread that
// processes its stream.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "orbbec_camera/gpu/sdk_algorithms.h"

namespace orbbec_camera {
namespace gpu {

class DepthPipeline {
 public:
  // allow_cuda false forces the CPU references. description names the
  // backend chosen and, for a fallback, why.
  static std::unique_ptr<DepthPipeline> create(bool allow_cuda, std::string *description);
  virtual ~DepthPipeline() = default;

  virtual bool usesGpu() const = 0;
  virtual void setSpeckle(const SpeckleParams &params) = 0;
  virtual bool setAlignment(const D2CConfig &config, std::string *error) = 0;
  virtual bool hasAlignment() const = 0;
  // See CudaDepthStage::process.
  virtual bool process(const uint16_t *raw, int width, int height, bool speckle,
                       uint16_t *filtered_out, uint16_t *aligned_out, std::string *error) = 0;
};

class ColorPipeline {
 public:
  enum class Backend { kNvHardware, kCudaRemap, kCpu };

  static std::unique_ptr<ColorPipeline> create(bool allow_cuda, bool allow_hw_codec,
                                               std::string *description);
  virtual ~ColorPipeline() = default;

  virtual Backend backend() const = 0;
  // Sets the undistortion of the next frames; the LUT's size is the frame's.
  virtual bool configure(const UndistortLut &lut, std::string *error) = 0;
  // Decodes one JPEG and rectifies it. rgb_out (width * height * 3 bytes)
  // receives the rectified RGB, jpeg_out a JPEG of it at jpeg_quality; at
  // least one of them is non-null.
  virtual bool process(const uint8_t *jpeg, size_t size, uint8_t *rgb_out,
                       std::vector<uint8_t> *jpeg_out, int jpeg_quality,
                       std::string *error) = 0;
};

const char *backendName(ColorPipeline::Backend backend);

// libjpeg: the codec of the CPU backends, and the encoder the driver uses
// for rectified colour when the GPU pipeline is off. The decode is the
// SDK's, bit for bit.
bool decodeJpegRgbCpu(const uint8_t *jpeg, size_t size, int width, int height,
                      std::vector<uint8_t> *rgb, std::string *error);
bool encodeRgbJpegCpu(const uint8_t *rgb, int width, int height, int quality,
                      std::vector<uint8_t> *jpeg, std::string *error);

}  // namespace gpu
}  // namespace orbbec_camera
