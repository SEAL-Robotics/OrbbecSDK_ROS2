// Copyright 2026 SEAL Robotics

#include "orbbec_camera/gpu/pipelines.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <setjmp.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <utility>

// After the C++ headers: libjpeg's header needs FILE and size_t.
#include <jpeglib.h>

#if defined(ORBBEC_WITH_CUDA)
#include "orbbec_camera/gpu/cuda_stages.h"
#endif
#if defined(ORBBEC_WITH_NVJPG)
#include "nv_jpeg.h"
#endif

namespace orbbec_camera {
namespace gpu {
namespace {

class CpuDepthPipeline : public DepthPipeline {
 public:
  bool usesGpu() const override { return false; }
  void setSpeckle(const SpeckleParams &params) override { speckle_ = params; }
  bool setAlignment(const D2CConfig &config, std::string *) override {
    config_ = config;
    tables_ = buildD2CTables(config);
    has_alignment_ = true;
    return true;
  }
  bool hasAlignment() const override { return has_alignment_; }
  bool process(const uint16_t *raw, int width, int height, bool speckle, uint16_t *filtered_out,
               uint16_t *aligned_out, std::string *error) override {
    if (aligned_out && (!has_alignment_ || width != tables_.width || height != tables_.height)) {
      if (error) {
        *error = "alignment not set up for this depth size";
      }
      return false;
    }
    const uint16_t *depth = raw;
    if (speckle) {
      filtered_.resize(static_cast<size_t>(width) * height);
      speckleFilterCpu(raw, filtered_.data(), width, height, speckle_);
      depth = filtered_.data();
    }
    if (filtered_out) {
      std::copy(depth, depth + static_cast<size_t>(width) * height, filtered_out);
    }
    if (aligned_out) {
      alignDepthToColorCpu(depth, aligned_out, config_, tables_);
    }
    return true;
  }

 private:
  SpeckleParams speckle_;
  D2CConfig config_;
  D2CTables tables_;
  bool has_alignment_ = false;
  std::vector<uint16_t> filtered_;
};

#if defined(ORBBEC_WITH_CUDA)
class CudaDepthPipeline : public DepthPipeline {
 public:
  bool usesGpu() const override { return true; }
  void setSpeckle(const SpeckleParams &params) override { stage_.setSpeckle(params); }
  bool setAlignment(const D2CConfig &config, std::string *error) override {
    return stage_.setAlignment(config, error);
  }
  bool hasAlignment() const override { return stage_.hasAlignment(); }
  bool process(const uint16_t *raw, int width, int height, bool speckle, uint16_t *filtered_out,
               uint16_t *aligned_out, std::string *error) override {
    return stage_.process(raw, width, height, speckle, filtered_out, aligned_out, error);
  }

 private:
  CudaDepthStage stage_;
};
#endif

// Decode and encode on the CPU; the remap on CUDA when there is one.
class CpuCodecColorPipeline : public ColorPipeline {
 public:
  explicit CpuCodecColorPipeline(bool use_cuda) : use_cuda_(use_cuda) {}

  Backend backend() const override { return use_cuda_ ? Backend::kCudaRemap : Backend::kCpu; }

  bool configure(const UndistortLut &lut, std::string *error) override {
    lut_ = lut;
#if defined(ORBBEC_WITH_CUDA)
    if (use_cuda_) {
      return stage_.setLut(lut, error);
    }
#else
    (void)error;
#endif
    return true;
  }

  bool process(const uint8_t *jpeg, size_t size, uint8_t *rgb_out, std::vector<uint8_t> *jpeg_out,
               int jpeg_quality, std::string *error) override {
    if (!decodeJpegRgbCpu(jpeg, size, lut_.width, lut_.height, &decoded_, error)) {
      return false;
    }
    uint8_t *rectified = rgb_out;
    if (rectified == nullptr) {
      rectified_.resize(decoded_.size());
      rectified = rectified_.data();
    }
    const size_t pitch = static_cast<size_t>(lut_.width) * 3;
#if defined(ORBBEC_WITH_CUDA)
    if (use_cuda_) {
      if (!stage_.rectifyHost(decoded_.data(), 3, pitch, rectified, nullptr, error)) {
        return false;
      }
    } else {
      remapRgbCpu(decoded_.data(), 3, pitch, rectified, lut_);
    }
#else
    remapRgbCpu(decoded_.data(), 3, pitch, rectified, lut_);
#endif
    if (jpeg_out) {
      return encodeRgbJpegCpu(rectified, lut_.width, lut_.height, jpeg_quality, jpeg_out, error);
    }
    return true;
  }

 private:
  bool use_cuda_;
  UndistortLut lut_;
  std::vector<uint8_t> decoded_;
  std::vector<uint8_t> rectified_;
#if defined(ORBBEC_WITH_CUDA)
  CudaColorStage stage_;
#endif
};

#if defined(ORBBEC_WITH_NVJPG)
// NVJPG decode, CUDA conversion to RGB and remap, and NVJPG encode of the
// rectified frame from a YCbCr surface the remap writes.
class NvColorPipeline : public ColorPipeline {
 public:
  Backend backend() const override { return Backend::kNvHardware; }

  bool configure(const UndistortLut &lut, std::string *error) override {
    if (!hardware_ || width_ != lut.width || height_ != lut.height) {
      hardware_ = NvJpegHardware::create(lut.width, lut.height, error);
      if (!hardware_) {
        return false;
      }
      width_ = lut.width;
      height_ = lut.height;
    }
    return stage_.setLut(lut, error);
  }

  bool process(const uint8_t *jpeg, size_t size, uint8_t *rgb_out, std::vector<uint8_t> *jpeg_out,
               int jpeg_quality, std::string *error) override {
    if (!hardware_) {
      if (error) {
        *error = "process called before configure";
      }
      return false;
    }
    YcbcrPlanes planes;
    if (!hardware_->decode(jpeg, size, &planes, error)) {
      return false;
    }
    const Yuv420Planes *yuv = jpeg_out ? &hardware_->encodeInput() : nullptr;
    const bool rectified = stage_.rectifyYcbcr(planes, rgb_out, yuv, error);
    hardware_->releaseDecoded();
    if (!rectified) {
      return false;
    }
    if (jpeg_out) {
      return hardware_->encode(jpeg_quality, jpeg_out, error);
    }
    return true;
  }

 private:
  int width_ = 0;
  int height_ = 0;
  std::unique_ptr<NvJpegHardware> hardware_;
  CudaColorStage stage_;
};

// Whether the hardware path works here: the engines open and the surfaces
// map for CUDA. Probed on a small frame so a failure is known up front.
bool nvHardwareUsable(std::string *reason) {
  return NvJpegHardware::create(64, 64, reason) != nullptr;
}
#endif

}  // namespace

const char *backendName(ColorPipeline::Backend backend) {
  switch (backend) {
    case ColorPipeline::Backend::kNvHardware:
      return "NVJPG decode/encode, CUDA colour conversion and remap";
    case ColorPipeline::Backend::kCudaRemap:
      return "CPU libjpeg decode/encode, CUDA remap";
    case ColorPipeline::Backend::kCpu:
      return "CPU libjpeg decode/encode, CPU remap";
  }
  return "unknown";
}

std::unique_ptr<DepthPipeline> DepthPipeline::create(bool allow_cuda, std::string *description) {
  std::string reason = "disabled";
#if defined(ORBBEC_WITH_CUDA)
  if (allow_cuda && cudaDeviceAvailable(&reason)) {
    if (description) {
      *description = "CUDA speckle filter and alignment";
    }
    return std::make_unique<CudaDepthPipeline>();
  }
#else
  (void)allow_cuda;
  reason = "built without CUDA";
#endif
  if (description) {
    *description = "CPU speckle filter and alignment (" + reason + ")";
  }
  return std::make_unique<CpuDepthPipeline>();
}

std::unique_ptr<ColorPipeline> ColorPipeline::create(bool allow_cuda, bool allow_hw_codec,
                                                     std::string *description) {
  std::string reason = "disabled";
  bool cuda = false;
#if defined(ORBBEC_WITH_CUDA)
  cuda = allow_cuda && cudaDeviceAvailable(&reason);
#else
  (void)allow_cuda;
  reason = "built without CUDA";
#endif
  std::unique_ptr<ColorPipeline> pipeline;
#if defined(ORBBEC_WITH_NVJPG)
  std::string hw_reason = "disabled";
  if (cuda && allow_hw_codec && nvHardwareUsable(&hw_reason)) {
    pipeline = std::make_unique<NvColorPipeline>();
  } else if (cuda) {
    reason = "NVJPG unavailable: " + hw_reason;
  }
#else
  (void)allow_hw_codec;
  if (cuda) {
    reason = "built without the Jetson Multimedia API";
  }
#endif
  if (!pipeline) {
    pipeline = std::make_unique<CpuCodecColorPipeline>(cuda);
  }
  if (description) {
    std::ostringstream text;
    text << backendName(pipeline->backend());
    if (pipeline->backend() != Backend::kNvHardware) {
      text << " (" << reason << ")";
    }
    *description = text.str();
  }
  return pipeline;
}

namespace {

struct JpegErrorManager {
  jpeg_error_mgr base;
  jmp_buf jump_buffer;
};

void jpegErrorExit(j_common_ptr cinfo) {
  longjmp(reinterpret_cast<JpegErrorManager *>(cinfo->err)->jump_buffer, 1);
}

void jpegQuiet(j_common_ptr, int) {}

}  // namespace

// The SDK's MJPG decode, reproduced: libjpeg with the fast integer inverse
// DCT and without fancy upsampling (test_pipelines pins it to the SDK).
bool decodeJpegRgbCpu(const uint8_t *jpeg, size_t size, int width, int height,
                      std::vector<uint8_t> *rgb, std::string *error) {
  jpeg_decompress_struct cinfo{};
  JpegErrorManager jerr{};
  cinfo.err = jpeg_std_error(&jerr.base);
  jerr.base.error_exit = jpegErrorExit;
  jerr.base.emit_message = jpegQuiet;
  if (setjmp(jerr.jump_buffer) != 0) {
    jpeg_destroy_decompress(&cinfo);
    if (error) {
      *error = "libjpeg could not decode the frame";
    }
    return false;
  }
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, jpeg, static_cast<unsigned long>(size));
  jpeg_read_header(&cinfo, TRUE);
  if (static_cast<int>(cinfo.image_width) != width ||
      static_cast<int>(cinfo.image_height) != height) {
    if (error) {
      std::ostringstream message;
      message << "JPEG is " << cinfo.image_width << "x" << cinfo.image_height << ", expected "
              << width << "x" << height;
      *error = message.str();
    }
    jpeg_destroy_decompress(&cinfo);
    return false;
  }
  cinfo.out_color_space = JCS_RGB;
  cinfo.dct_method = JDCT_IFAST;
  cinfo.do_fancy_upsampling = FALSE;
  jpeg_start_decompress(&cinfo);
  rgb->resize(static_cast<size_t>(width) * height * 3);
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row = rgb->data() + static_cast<size_t>(cinfo.output_scanline) * width * 3;
    jpeg_read_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return true;
}

bool encodeRgbJpegCpu(const uint8_t *rgb, int width, int height, int quality,
                      std::vector<uint8_t> *jpeg, std::string *error) {
  const cv::Mat in(height, width, CV_8UC3, const_cast<uint8_t *>(rgb));
  cv::Mat bgr;
  cv::cvtColor(in, bgr, cv::COLOR_RGB2BGR);
  if (!cv::imencode(".jpg", bgr, *jpeg, {cv::IMWRITE_JPEG_QUALITY, quality})) {
    if (error) {
      *error = "libjpeg could not encode the frame";
    }
    return false;
  }
  return true;
}

}  // namespace gpu
}  // namespace orbbec_camera
