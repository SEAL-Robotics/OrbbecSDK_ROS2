// Copyright 2026 SEAL Robotics

#include "orbbec_camera/gpu_image_pipeline.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

#if defined(ORBBEC_WITH_CUDA)
#include "orbbec_camera/gpu/cuda_stages.h"
#endif

namespace orbbec_camera {
namespace {

constexpr double kSummaryPeriodS = 10.0;

std::shared_ptr<ob::Frame> frameFromProfile(const std::shared_ptr<ob::StreamProfile> &profile,
                                            const std::shared_ptr<ob::Frame> &info_source) {
  auto frame = ob::FrameFactory::createFrameFromStreamProfile(profile);
  frame->copyFrameInfo(info_source);
  return frame;
}

std::shared_ptr<ob::Frame> alignedDepthOf(const std::shared_ptr<ob::Frame> &result) {
  if (!result) {
    return nullptr;
  }
  if (result->is<ob::FrameSet>()) {
    return result->as<ob::FrameSet>()->getFrame(OB_FRAME_DEPTH);
  }
  return result->getType() == OB_FRAME_DEPTH ? result : nullptr;
}

std::shared_ptr<ob::Frame> sdkAlign(ob::Align &align, const std::shared_ptr<ob::Frame> &depth,
                                    const std::shared_ptr<ob::Frame> &color) {
  auto frame_set = ob::FrameFactory::createFrameSet();
  frame_set->pushFrame(depth);
  frame_set->pushFrame(color);
  return alignedDepthOf(align.process(frame_set));
}

template <typename T>
void appendBytes(std::string *key, const T &value) {
  key->append(reinterpret_cast<const char *>(&value), sizeof(T));
}

bool distortionIsTrivial(const OBCameraDistortion &d) {
  return d.model == OB_DISTORTION_NONE ||
         (d.k1 == 0.0f && d.k2 == 0.0f && d.k3 == 0.0f && d.k4 == 0.0f && d.k5 == 0.0f &&
          d.k6 == 0.0f && d.p1 == 0.0f && d.p2 == 0.0f);
}

// The camera pads its JPEG with zeros after the end-of-image marker, which
// the hardware decoder does not skip as libjpeg does.
size_t jpegSize(const std::shared_ptr<ob::Frame> &frame) {
  const uint8_t *data = frame->getData();
  size_t size = frame->getDataSize();
  while (size > 4 && data[size - 1] == 0x00) {
    --size;
  }
  return size;
}

}  // namespace

struct GpuImagePipeline::StageStats {
  struct Value {
    uint64_t count = 0;
    double sum = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
  };
  uint64_t frames = 0;
  std::map<std::string, Value> values;
};

bool GpuImagePipeline::available(std::string *reason) {
#if defined(ORBBEC_WITH_CUDA)
  return gpu::cudaDeviceAvailable(reason);
#else
  *reason = "this orbbec_camera was built without CUDA";
  return false;
#endif
}

GpuImagePipeline::GpuImagePipeline(rclcpp::Node *node, const Options &options)
    : node_(node), logger_(node->get_logger().get_child("gpu_pipeline")), options_(options) {
  options_.compare_every_n = std::max(1, options_.compare_every_n);
  setSpeckle(true, options_.speckle.max_diff, options_.speckle.max_size);
  RCLCPP_INFO_STREAM(logger_, "GPU image pipeline: speckle filter "
                                  << (options_.speckle_filter ? "on" : "off (SDK)")
                                  << " (max_diff " << options_.speckle.max_diff << ", max_size "
                                  << options_.speckle.max_size << "), depth alignment "
                                  << (options_.depth_alignment ? "on" : "off (SDK)")
                                  << ", colour decode/undistortion/JPEG "
                                  << (options_.color ? "on" : "off (SDK)") << ", comparison "
                                  << (options_.compare ? "every " +
                                                             std::to_string(options_.compare_every_n) +
                                                             " frames"
                                                       : std::string("off")));
  if (options_.compare) {
    comparison_pub_ = node_->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "gpu_pipeline/comparison", rclcpp::QoS(10));
    summary_timer_ = node_->create_wall_timer(
        std::chrono::duration<double>(kSummaryPeriodS), [this]() {
          std::lock_guard<std::mutex> lock(stats_mutex_);
          for (const auto &stage : stats_) {
            std::ostringstream line;
            line << "SDK vs GPU, " << stage.first << ": " << stage.second->frames << " frames";
            for (const auto &value : stage.second->values) {
              const auto &v = value.second;
              line << ", " << value.first << " min " << v.min << " mean "
                   << (v.count ? v.sum / static_cast<double>(v.count) : 0.0) << " max " << v.max;
            }
            RCLCPP_INFO_STREAM(logger_, line.str());
          }
        });
  }
}

GpuImagePipeline::~GpuImagePipeline() {
  // The timer's callback holds this object.
  if (summary_timer_) {
    summary_timer_->cancel();
    summary_timer_.reset();
  }
}

void GpuImagePipeline::logFailure(const char *stage, const std::string &error) {
  RCLCPP_ERROR_STREAM_THROTTLE(logger_, *node_->get_clock(), 5000,
                               "GPU " << stage << " failed, the SDK does this frame: " << error);
}

// ---------------------------------------------------------------------------
// Depth

void GpuImagePipeline::setSpeckle(bool enabled, int max_diff, int max_size) {
  speckle_max_diff_.store(max_diff);
  speckle_max_size_.store(max_size);
  speckle_enabled_.store(enabled);
}

void GpuImagePipeline::startDepthFrame() {
  compare_depth_now_ = options_.compare && depth_frames_ % options_.compare_every_n == 0;
  ++depth_frames_;
  compare_sdk_filtered_.reset();
}

void GpuImagePipeline::ensureDepthPipeline() {
  if (!depth_) {
    std::string description;
    depth_ = gpu::DepthPipeline::create(true, &description);
    RCLCPP_INFO_STREAM(logger_, "depth: " << description);
  }
}

std::shared_ptr<ob::Frame> GpuImagePipeline::filterDepth(const std::shared_ptr<ob::Frame> &raw,
                                                         bool data_wanted) {
  ensureDepthPipeline();
  auto video = raw->as<ob::VideoFrame>();
  const int width = static_cast<int>(video->getWidth());
  const int height = static_cast<int>(video->getHeight());
  const size_t pixels = static_cast<size_t>(width) * height;
  if (raw->getFormat() != OB_FORMAT_Y16 && raw->getFormat() != OB_FORMAT_Z16) {
    logFailure("speckle filter", "depth format is not 16-bit");
    return nullptr;
  }
  auto filtered = ob::FrameFactory::createFrameFromOtherFrame(raw, false);
  if (raw->getDataSize() < pixels * sizeof(uint16_t) ||
      filtered->getDataSize() < pixels * sizeof(uint16_t)) {
    logFailure("speckle filter", "depth frame buffer is smaller than its resolution");
    return nullptr;
  }
  uint16_t *out = reinterpret_cast<uint16_t *>(filtered->getData());
  if (!data_wanted && !compare_depth_now_) {
    std::memset(out, 0, pixels * sizeof(uint16_t));
    return filtered;
  }
  depth_->setSpeckle({speckle_max_diff_.load(), speckle_max_size_.load()});
  std::string error;
  if (!depth_->process(reinterpret_cast<const uint16_t *>(raw->getData()), width, height, true,
                       out, nullptr, &error)) {
    logFailure("speckle filter", error);
    return nullptr;
  }
  if (compare_depth_now_) {
    compareSpeckle(raw, out);
  }
  return filtered;
}

bool GpuImagePipeline::ensureDepthConfig(const std::shared_ptr<ob::Frame> &depth,
                                         const std::shared_ptr<ob::Frame> &color,
                                         ob::Align &sdk_align) {
  auto depth_profile = depth->getStreamProfile()->as<ob::VideoStreamProfile>();
  auto color_profile = color->getStreamProfile()->as<ob::VideoStreamProfile>();
  gpu::D2CConfig config;
  config.depth_intrinsic = depth_profile->getIntrinsic();
  config.depth_distortion = depth_profile->getDistortion();
  config.color_intrinsic = color_profile->getIntrinsic();
  config.depth_to_color = depth_profile->getExtrinsicTo(color_profile);
  config.depth_unit_mm = depth->as<ob::DepthFrame>()->getValueScale();
  std::string key;
  appendBytes(&key, config.depth_intrinsic);
  appendBytes(&key, config.depth_distortion);
  appendBytes(&key, config.color_intrinsic);
  appendBytes(&key, config.depth_to_color);
  appendBytes(&key, config.depth_unit_mm);
  appendBytes(&key, depth_profile->getWidth());
  appendBytes(&key, color_profile->getWidth());
  if (key == depth_key_ && aligned_profile_ && depth_->hasAlignment()) {
    return true;
  }
  depth_key_.clear();

  // The output profile is whatever the SDK's own alignment gives its output.
  auto sdk_aligned = sdkAlign(sdk_align, depth, color);
  if (!sdk_aligned) {
    logFailure("depth alignment", "the SDK alignment gave no frame to take the profile from");
    return false;
  }
  aligned_profile_ = sdk_aligned->getStreamProfile();
  {
    auto aligned_video = sdk_aligned->as<ob::VideoFrame>();
    if (aligned_video->getWidth() != static_cast<uint32_t>(config.color_intrinsic.width) ||
        aligned_video->getHeight() != static_cast<uint32_t>(config.color_intrinsic.height) ||
        depth_profile->getWidth() != static_cast<uint32_t>(config.depth_intrinsic.width) ||
        depth_profile->getHeight() != static_cast<uint32_t>(config.depth_intrinsic.height)) {
      logFailure("depth alignment", "the intrinsics are not for the streams' resolutions");
      aligned_profile_.reset();
      return false;
    }
    aligned_bytes_ = static_cast<size_t>(config.color_intrinsic.width) *
                     config.color_intrinsic.height * sizeof(uint16_t);
  }

  // The SDK drops raw depth at or above the frame's maximum valid value,
  // which it does not expose: find it by aligning constant frames.
  auto probe = ob::FrameFactory::createFrameFromOtherFrame(depth, false);
  const size_t pixels = probe->getDataSize() / sizeof(uint16_t);
  auto survives = [&](uint32_t value) {
    auto *data = reinterpret_cast<uint16_t *>(probe->getData());
    std::fill(data, data + pixels, static_cast<uint16_t>(value));
    auto out = sdkAlign(sdk_align, probe, color);
    if (!out) {
      return false;
    }
    const auto *aligned = reinterpret_cast<const uint16_t *>(out->getData());
    const size_t n = out->getDataSize() / sizeof(uint16_t);
    return std::any_of(aligned, aligned + n, [](uint16_t v) { return v != 0; });
  };
  uint32_t max_valid = 65535;
  if (!survives(1000)) {
    RCLCPP_WARN_STREAM(logger_, "could not probe the SDK's maximum valid depth; using 65535");
  } else if (!survives(65535)) {
    uint32_t low = 1000, high = 65535;
    while (high - low > 1) {
      const uint32_t mid = (low + high) / 2;
      (survives(mid) ? low : high) = mid;
    }
    max_valid = high;
  }
  config.max_valid_depth = static_cast<uint16_t>(max_valid);
  config.gap_fill_copy = gpu::sdkGapFillCopy(config.depth_intrinsic, config.color_intrinsic);

  std::string error;
  if (!depth_->setAlignment(config, &error)) {
    logFailure("depth alignment", error);
    return false;
  }
  depth_key_ = key;
  RCLCPP_INFO_STREAM(logger_, "depth alignment set up: depth "
                                  << config.depth_intrinsic.width << "x"
                                  << config.depth_intrinsic.height << " to colour "
                                  << config.color_intrinsic.width << "x"
                                  << config.color_intrinsic.height << ", depth unit "
                                  << config.depth_unit_mm << " mm, raw depth valid below "
                                  << config.max_valid_depth << ", gap fill "
                                  << (config.gap_fill_copy ? "2x2 copy" : "rectangle"));
  return true;
}

std::shared_ptr<ob::Frame> GpuImagePipeline::alignDepth(const std::shared_ptr<ob::Frame> &depth,
                                                        const std::shared_ptr<ob::Frame> &color,
                                                        ob::Align &sdk_align, bool data_wanted) {
  ensureDepthPipeline();
  const bool compare_now = compare_depth_now_;
  if (!ensureDepthConfig(depth, color, sdk_align)) {
    return nullptr;
  }
  auto aligned = frameFromProfile(aligned_profile_, depth);
  auto video = depth->as<ob::VideoFrame>();
  if (aligned->getDataSize() < aligned_bytes_ ||
      depth->getDataSize() <
          static_cast<size_t>(video->getWidth()) * video->getHeight() * sizeof(uint16_t)) {
    logFailure("depth alignment", "a frame buffer is smaller than its resolution");
    return nullptr;
  }
  auto *out = reinterpret_cast<uint16_t *>(aligned->getData());
  if (!data_wanted && !compare_now) {
    std::memset(out, 0, aligned->getDataSize());
    return aligned;
  }
  std::string error;
  if (!depth_->process(reinterpret_cast<const uint16_t *>(depth->getData()),
                       static_cast<int>(video->getWidth()), static_cast<int>(video->getHeight()),
                       false, nullptr, out, &error)) {
    logFailure("depth alignment", error);
    return nullptr;
  }
  if (compare_now) {
    compareAlignment(depth, color, out, sdk_align);
  }
  return aligned;
}

void GpuImagePipeline::compareSpeckle(const std::shared_ptr<ob::Frame> &raw,
                                      const uint16_t *gpu_filtered) {
  if (!sdk_noise_filter_tried_) {
    sdk_noise_filter_tried_ = true;
    try {
      auto filter = ob::FilterFactory::createFilter("NoiseRemovalFilter");
      for (const auto &item : filter->getConfigSchemaVec()) {
        const std::string name = item.name;
        if (name == "max_size") {
          filter->setConfigValue(name, speckle_max_size_.load());
        } else if (name == "min_diff") {
          filter->setConfigValue(name, speckle_max_diff_.load());
        }
      }
      filter->enable(true);
      sdk_noise_filter_ = filter;
      RCLCPP_INFO_STREAM(logger_, "comparing the speckle filter with the SDK's NoiseRemovalFilter");
    } catch (const std::exception &e) {
      RCLCPP_WARN_STREAM(logger_, "the SDK's NoiseRemovalFilter is unavailable here ("
                                      << e.what()
                                      << "): the speckle filter is compared through the "
                                         "alignment only");
    }
  }
  if (!sdk_noise_filter_) {
    return;
  }
  std::shared_ptr<ob::Frame> sdk;
  try {
    sdk = sdk_noise_filter_->process(raw);
  } catch (const std::exception &e) {
    RCLCPP_WARN_STREAM_THROTTLE(logger_, *node_->get_clock(), 10000,
                                "SDK NoiseRemovalFilter failed: " << e.what());
    return;
  }
  if (!sdk || sdk->getDataSize() != raw->getDataSize()) {
    return;
  }
  const size_t pixels = raw->getDataSize() / sizeof(uint16_t);
  const auto *raw_data = reinterpret_cast<const uint16_t *>(raw->getData());
  const auto *sdk_data = reinterpret_cast<const uint16_t *>(sdk->getData());
  const auto diff = gpu::compareDepth(sdk_data, gpu_filtered, pixels);
  const size_t valid_raw = gpu::compareDepth(raw_data, raw_data, pixels).valid_a;
  record("speckle_filter",
         {{"valid_mask_iou", diff.validIoU()},
          {"removed_by_sdk", static_cast<double>(valid_raw - diff.valid_a)},
          {"removed_by_gpu", static_cast<double>(valid_raw - diff.valid_b)},
          {"valid_only_sdk", static_cast<double>(diff.valid_either - diff.valid_b)},
          {"valid_only_gpu", static_cast<double>(diff.valid_either - diff.valid_a)},
          {"max_abs_diff", static_cast<double>(diff.max_abs_diff)}});
  compare_sdk_filtered_ = sdk;
}

void GpuImagePipeline::compareAlignment(const std::shared_ptr<ob::Frame> &gpu_filtered,
                                        const std::shared_ptr<ob::Frame> &color,
                                        const uint16_t *gpu_aligned, ob::Align &sdk_align) {
  auto sdk = sdkAlign(sdk_align, gpu_filtered, color);
  if (sdk) {
    const size_t pixels = sdk->getDataSize() / sizeof(uint16_t);
    const auto diff =
        gpu::compareDepth(reinterpret_cast<const uint16_t *>(sdk->getData()), gpu_aligned, pixels);
    record("depth_alignment", {{"valid_mask_iou", diff.validIoU()},
                               {"differing_pixels", static_cast<double>(diff.differing)},
                               {"mean_abs_diff", diff.mean_abs_diff},
                               {"max_abs_diff", static_cast<double>(diff.max_abs_diff)}});
  }
  // SDK filter then SDK alignment against the GPU's two stages.
  if (compare_sdk_filtered_) {
    auto sdk_both = sdkAlign(sdk_align, compare_sdk_filtered_, color);
    compare_sdk_filtered_.reset();
    if (sdk_both) {
      const size_t pixels = sdk_both->getDataSize() / sizeof(uint16_t);
      const auto diff = gpu::compareDepth(reinterpret_cast<const uint16_t *>(sdk_both->getData()),
                                          gpu_aligned, pixels);
      record("aligned_depth", {{"valid_mask_iou", diff.validIoU()},
                               {"mean_abs_diff", diff.mean_abs_diff},
                               {"max_abs_diff", static_cast<double>(diff.max_abs_diff)}});
    }
  }
}

// ---------------------------------------------------------------------------
// Colour

bool GpuImagePipeline::ensureColorConfig(const std::shared_ptr<ob::Frame> &mjpg) {
  auto profile = mjpg->getStreamProfile()->as<ob::VideoStreamProfile>();
  const OBCameraIntrinsic intrinsic = profile->getIntrinsic();
  const OBCameraDistortion distortion = profile->getDistortion();
  std::string key;
  appendBytes(&key, intrinsic);
  appendBytes(&key, distortion);
  appendBytes(&key, profile->getWidth());
  appendBytes(&key, profile->getHeight());
  if (key == color_key_ && rectified_profile_) {
    return true;
  }
  color_key_.clear();
  if (distortionIsTrivial(distortion)) {
    logFailure("colour", "the colour stream has no distortion; the SDK passes it through");
    return false;
  }
  if (static_cast<uint32_t>(intrinsic.width) != profile->getWidth() ||
      static_cast<uint32_t>(intrinsic.height) != profile->getHeight()) {
    logFailure("colour", "the colour intrinsics are not for the stream's resolution");
    return false;
  }
  // The output profile is whatever the SDK's own undistortion gives its
  // output.
  if (!sdk_undistortion_) {
    sdk_undistortion_ = std::make_unique<ob::UnDistortionFilter>(OB_STREAM_COLOR);
    sdk_undistortion_->enable(true);
  }
  auto sdk = sdk_undistortion_->process(mjpg);
  if (!sdk) {
    logFailure("colour", "the SDK undistortion gave no frame to take the profile from");
    return false;
  }
  rectified_profile_ = sdk->getStreamProfile();
  rectified_bytes_ = static_cast<size_t>(intrinsic.width) * intrinsic.height * 3;
  if (sdk->getFormat() != OB_FORMAT_RGB || sdk->getDataSize() < rectified_bytes_) {
    logFailure("colour", "the SDK's undistortion does not give RGB of the stream's size");
    rectified_profile_.reset();
    return false;
  }
  std::string error;
  if (!color_->configure(gpu::buildUndistortLut(intrinsic, distortion), &error)) {
    logFailure("colour", error);
    rectified_profile_.reset();
    return false;
  }
  color_key_ = key;
  RCLCPP_INFO_STREAM(logger_, "colour undistortion set up for " << intrinsic.width << "x"
                                                               << intrinsic.height);
  return true;
}

bool GpuImagePipeline::processColor(const std::shared_ptr<ob::Frame> &mjpg, bool rgb_wanted,
                                    bool jpeg_wanted, std::shared_ptr<ob::Frame> *rectified,
                                    std::vector<uint8_t> *jpeg) {
  if (!color_) {
    std::string description;
    color_ = gpu::ColorPipeline::create(true, true, &description);
    RCLCPP_INFO_STREAM(logger_, "colour: " << description);
  }
  const bool compare_now = options_.compare && color_frames_ % options_.compare_every_n == 0;
  ++color_frames_;
  if (!ensureColorConfig(mjpg)) {
    return false;
  }
  auto out = frameFromProfile(rectified_profile_, mjpg);
  if (out->getDataSize() < rectified_bytes_) {
    logFailure("colour", "the rectified frame buffer is smaller than its resolution");
    return false;
  }
  uint8_t *rgb = (rgb_wanted || compare_now) ? out->getData() : nullptr;
  if (rgb == nullptr) {
    // Nothing reads the pixels now, but a reader that subscribes before the
    // publish gets a black frame, never stale memory.
    std::memset(out->getData(), 0, out->getDataSize());
  }
  if (rgb != nullptr || jpeg_wanted) {
    std::string error;
    if (!color_->process(mjpg->getData(), jpegSize(mjpg), rgb, jpeg_wanted ? jpeg : nullptr,
                         options_.jpeg_quality, &error)) {
      logFailure("colour", error);
      return false;
    }
  }
  if (compare_now) {
    compareColor(mjpg, rgb);
  }
  *rectified = out;
  return true;
}

std::shared_ptr<ob::Frame> GpuImagePipeline::sdkRectify(const std::shared_ptr<ob::Frame> &mjpg) {
  if (!sdk_undistortion_) {
    sdk_undistortion_ = std::make_unique<ob::UnDistortionFilter>(OB_STREAM_COLOR);
    sdk_undistortion_->enable(true);
  }
  try {
    return sdk_undistortion_->process(mjpg);
  } catch (const std::exception &e) {
    logFailure("colour fallback", e.what());
    return nullptr;
  }
}

void GpuImagePipeline::compareColor(const std::shared_ptr<ob::Frame> &mjpg,
                                    const uint8_t *gpu_rgb) {
  auto sdk = sdk_undistortion_->process(mjpg);
  if (!sdk) {
    return;
  }
  const auto diff = gpu::compareColor(sdk->getData(), gpu_rgb, sdk->getDataSize());
  record("color", {{"psnr_db", std::isinf(diff.psnr_db) ? 999.0 : diff.psnr_db},
                   {"mean_abs_diff", diff.mean_abs_diff},
                   {"max_abs_diff", static_cast<double>(diff.max_abs_diff)}});
}

// ---------------------------------------------------------------------------

void GpuImagePipeline::record(const std::string &stage,
                              const std::map<std::string, double> &values) {
  diagnostic_msgs::msg::DiagnosticArray message;
  message.header.stamp = node_->now();
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.name = "gpu_pipeline/" + stage;
  status.message = "SDK CPU path vs GPU path, same frame";
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    auto &stats = stats_[stage];
    if (!stats) {
      stats = std::make_unique<StageStats>();
    }
    ++stats->frames;
    for (const auto &entry : values) {
      auto &v = stats->values[entry.first];
      ++v.count;
      v.sum += entry.second;
      v.min = std::min(v.min, entry.second);
      v.max = std::max(v.max, entry.second);
      diagnostic_msgs::msg::KeyValue key_value;
      key_value.key = entry.first;
      std::ostringstream text;
      text << entry.second;
      key_value.value = text.str();
      status.values.push_back(key_value);
    }
  }
  message.status.push_back(status);
  if (comparison_pub_) {
    comparison_pub_->publish(message);
  }
}

}  // namespace orbbec_camera
