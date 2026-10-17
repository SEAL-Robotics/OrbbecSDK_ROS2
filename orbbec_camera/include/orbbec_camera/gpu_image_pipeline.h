// Copyright 2026 SEAL Robotics
//
// The driver's side of the GPU image pipeline (gpu_pipeline:=true): the
// software noise filter, depth-to-colour alignment and colour MJPG decode
// and undistortion the SDK would run on the CPU, run instead by
// orbbec_camera::gpu, plus the NVJPG encode of the rectified colour for
// color/image_raw/compressed.
//
// Its outputs are SDK frames carrying the profiles and frame info the SDK
// itself would have produced (each profile is taken once from the SDK's own
// filter), so camera_info, frame ids, stamps and everything downstream in
// OBCameraNode stay as they were.
//
// With compare set, each stage also runs the SDK's CPU filter on the same
// input every compare_every_n frames, and publishes the differences on
// gpu_pipeline/comparison and in a periodic log line.
//
// The depth half runs on the SDK's frameset callback thread and the colour
// half on the driver's colour thread; neither is thread-safe on its own.
#pragma once

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "libobsensor/ObSensor.hpp"
#include "orbbec_camera/gpu/image_metrics.h"
#include "orbbec_camera/gpu/pipelines.h"

namespace orbbec_camera {

class GpuImagePipeline {
 public:
  struct Options {
    bool speckle_filter = true;
    bool depth_alignment = true;
    bool color = true;
    bool compare = false;
    int compare_every_n = 1;
    int jpeg_quality = 75;
    gpu::SpeckleParams speckle;
  };

  // Whether the GPU pipeline can run here at all; reason says why not.
  static bool available(std::string *reason);

  GpuImagePipeline(rclcpp::Node *node, const Options &options);
  ~GpuImagePipeline();

  // Whether the GPU owns the noise filter, fixed at start; the device's
  // own software filter stays off while it does.
  bool speckleFilterActive() const { return options_.speckle_filter; }
  // The filter's runtime switch and parameters (the SetFilter service and
  // filter parameters), taken from the next frame on.
  void setSpeckle(bool enabled, int max_diff, int max_size);
  bool speckleEnabled() const { return speckle_enabled_.load(); }
  bool depthAlignmentActive() const { return options_.depth_alignment; }
  bool colorActive() const { return options_.color; }
  bool compareActive() const { return options_.compare; }
  int jpegQuality() const { return options_.jpeg_quality; }

  // Call once per frameset before filterDepth/alignDepth: decides whether
  // this frame is one the comparison runs on.
  void startDepthFrame();

  // Speckle-filters raw depth into a new frame with raw's profile and info.
  // With data_wanted false nothing reads the pixels, so none are computed.
  // nullptr on failure, which is logged.
  std::shared_ptr<ob::Frame> filterDepth(const std::shared_ptr<ob::Frame> &raw, bool data_wanted);

  // Aligns depth to color's camera into a frame with the profile sdk_align
  // gives its output. nullptr on failure, which is logged; the caller falls
  // back to sdk_align.
  std::shared_ptr<ob::Frame> alignDepth(const std::shared_ptr<ob::Frame> &depth,
                                        const std::shared_ptr<ob::Frame> &color,
                                        ob::Align &sdk_align, bool data_wanted);

  // Decodes and undistorts one MJPG colour frame. rectified carries the
  // SDK undistortion's output profile and the MJPG frame's info, and its
  // pixels when rgb_wanted; jpeg receives the rectified image as JPEG when
  // jpeg_wanted. false on failure, which is logged.
  bool processColor(const std::shared_ptr<ob::Frame> &mjpg, bool rgb_wanted, bool jpeg_wanted,
                    std::shared_ptr<ob::Frame> *rectified, std::vector<uint8_t> *jpeg);
  // The SDK's own undistortion of one frame, for when processColor fails.
  std::shared_ptr<ob::Frame> sdkRectify(const std::shared_ptr<ob::Frame> &mjpg);

 private:
  struct StageStats;

  void logFailure(const char *stage, const std::string &error);
  void ensureDepthPipeline();
  bool ensureDepthConfig(const std::shared_ptr<ob::Frame> &depth,
                         const std::shared_ptr<ob::Frame> &color, ob::Align &sdk_align);
  bool ensureColorConfig(const std::shared_ptr<ob::Frame> &mjpg);
  void compareSpeckle(const std::shared_ptr<ob::Frame> &raw, const uint16_t *gpu_filtered);
  void compareAlignment(const std::shared_ptr<ob::Frame> &gpu_filtered,
                        const std::shared_ptr<ob::Frame> &color, const uint16_t *gpu_aligned,
                        ob::Align &sdk_align);
  void compareColor(const std::shared_ptr<ob::Frame> &mjpg, const uint8_t *gpu_rgb);
  void record(const std::string &stage, const std::map<std::string, double> &values);

  rclcpp::Node *node_;
  rclcpp::Logger logger_;
  Options options_;

  std::atomic<bool> speckle_enabled_{true};
  std::atomic<int> speckle_max_diff_{256};
  std::atomic<int> speckle_max_size_{200};

  // Depth thread.
  std::unique_ptr<gpu::DepthPipeline> depth_;
  bool compare_depth_now_ = false;
  std::string depth_key_;
  std::shared_ptr<ob::StreamProfile> aligned_profile_;
  int depth_frames_ = 0;
  std::shared_ptr<ob::Filter> sdk_noise_filter_;
  bool sdk_noise_filter_tried_ = false;
  std::shared_ptr<ob::Frame> compare_sdk_filtered_;
  std::vector<uint16_t> scratch_depth_;

  // Colour thread.
  std::unique_ptr<gpu::ColorPipeline> color_;
  std::string color_key_;
  std::shared_ptr<ob::StreamProfile> rectified_profile_;
  int color_frames_ = 0;
  std::unique_ptr<ob::UnDistortionFilter> sdk_undistortion_;

  // Comparison results, from both threads.
  std::mutex stats_mutex_;
  std::map<std::string, std::unique_ptr<StageStats>> stats_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr comparison_pub_;
  rclcpp::TimerBase::SharedPtr summary_timer_;
  std::map<std::string, rclcpp::Time> last_failure_log_;
};

}  // namespace orbbec_camera
