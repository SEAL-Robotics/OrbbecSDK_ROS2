// Copyright 2026 SEAL Robotics
//
// The CUDA kernels behind the driver's GPU pipeline: the depth speckle
// filter and depth-to-colour alignment, and for colour the conversion of a
// hardware-decoded JPEG to RGB, the undistortion remap, and the conversion
// back to YCbCr that feeds the hardware JPEG encoder. Every stage reproduces
// its CPU reference (sdk_algorithms.h, color_math.h) bit for bit.
//
// Host buffers are copied through pinned staging memory, and the caller's
// thread waits on a blocking-sync event rather than spinning while the GPU
// works. A stage is not thread-safe: give each thread its own.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "orbbec_camera/gpu/color_math.h"
#include "orbbec_camera/gpu/sdk_algorithms.h"

namespace orbbec_camera {
namespace gpu {

// Whether this build has CUDA and the machine a CUDA device; reason says
// why not when it returns false.
bool cudaDeviceAvailable(std::string *reason);

// An 8-bit image in GPU-visible memory.
struct DeviceImage {
  const uint8_t *data = nullptr;
  int channels = 0;
  size_t pitch = 0;
};

// Planes in GPU-visible memory that receive a JFIF YCbCr 4:2:0 image.
struct Yuv420Planes {
  uint8_t *plane[3] = {nullptr, nullptr, nullptr};
  size_t pitch[3] = {0, 0, 0};
};

class CudaDepthStage {
 public:
  CudaDepthStage();
  ~CudaDepthStage();
  CudaDepthStage(const CudaDepthStage &) = delete;
  CudaDepthStage &operator=(const CudaDepthStage &) = delete;

  void setSpeckle(const SpeckleParams &params);
  // Builds the alignment tables and uploads them; call when the profiles
  // or the extrinsic change.
  bool setAlignment(const D2CConfig &config, std::string *error);
  bool hasAlignment() const;

  // Speckle-filters raw (when speckle is set) and aligns the result to
  // colour (when aligned_out is non-null). filtered_out receives the
  // unaligned filtered depth when non-null.
  bool process(const uint16_t *raw, int width, int height, bool speckle, uint16_t *filtered_out,
               uint16_t *aligned_out, std::string *error);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class CudaColorStage {
 public:
  CudaColorStage();
  ~CudaColorStage();
  CudaColorStage(const CudaColorStage &) = delete;
  CudaColorStage &operator=(const CudaColorStage &) = delete;

  bool setLut(const UndistortLut &lut, std::string *error);

  // Rectifies src (lut.width x lut.height, 3 or 4 channels) into packed RGB
  // copied to rgb_out when non-null, and into yuv_out when non-null.
  bool rectify(const DeviceImage &src, uint8_t *rgb_out, const Yuv420Planes *yuv_out,
               std::string *error);
  // The same from a host image, which is uploaded first.
  bool rectifyHost(const uint8_t *src, int channels, size_t pitch, uint8_t *rgb_out,
                   const Yuv420Planes *yuv_out, std::string *error);
  // The same from decoded JPEG planes in GPU-visible memory, converted to
  // RGB first as libjpeg would.
  bool rectifyYcbcr(const YcbcrPlanes &src, uint8_t *rgb_out, const Yuv420Planes *yuv_out,
                    std::string *error);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu
}  // namespace orbbec_camera
