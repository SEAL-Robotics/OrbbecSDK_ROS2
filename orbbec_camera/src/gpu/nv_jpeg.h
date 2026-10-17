// Copyright 2026 SEAL Robotics
//
// Jetson NVJPG decode and encode around hardware surfaces CUDA reads and
// writes: decode hands out the decoder's YCbCr planes mapped for CUDA, and
// encode reads a YCbCr 4:2:0 surface the caller fills. Corrupt input fails
// the call instead of exiting the process, which is what libjpeg's default
// error handler would do.
//
// The planes are not converted on VIC: NVJPG labels its 4:2:2 output with
// BT.601's limited range, which has no full-range 4:2:2 counterpart to
// relabel it with, so VIC would stretch JPEG's full-range samples.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "orbbec_camera/gpu/cuda_stages.h"

namespace orbbec_camera {
namespace gpu {

class NvJpegHardware {
 public:
  // nullptr, with error set, when the engines or the CUDA interop are not
  // available.
  static std::unique_ptr<NvJpegHardware> create(int width, int height, std::string *error);
  ~NvJpegHardware();
  NvJpegHardware(const NvJpegHardware &) = delete;
  NvJpegHardware &operator=(const NvJpegHardware &) = delete;

  // planes stay mapped until releaseDecoded() or the next decode.
  bool decode(const uint8_t *jpeg, size_t size, YcbcrPlanes *planes, std::string *error);
  void releaseDecoded();
  // The surface encode() reads; fill it before each call.
  const Yuv420Planes &encodeInput() const;
  bool encode(int quality, std::vector<uint8_t> *jpeg, std::string *error);

 private:
  NvJpegHardware();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu
}  // namespace orbbec_camera
