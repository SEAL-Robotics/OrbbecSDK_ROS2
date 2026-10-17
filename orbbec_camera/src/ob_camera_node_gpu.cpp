// Copyright 2026 SEAL Robotics
//
// OBCameraNode's side of the GPU image pipeline (gpu_image_pipeline.h):
// deciding which stages run on the GPU, feeding them, and the driver's own
// JPEG of the rectified colour on color/image_raw/compressed.

#include <string>
#include <utility>
#include <vector>

#include "orbbec_camera/ob_camera_node.h"
#include "orbbec_camera/utils.h"

namespace orbbec_camera {

void OBCameraNode::resolveGpuPipeline() {
  gpu_speckle_filter_active_ = false;
  gpu_depth_alignment_active_ = false;
  gpu_color_active_ = false;
  if (!gpu_pipeline_) {
    return;
  }
  std::string reason;
  if (!GpuImagePipeline::available(&reason)) {
    RCLCPP_WARN_STREAM(logger_, "gpu_pipeline is set but cannot run (" << reason
                                                                       << "): the SDK does it all");
    return;
  }
  const bool depth = enable_stream_[DEPTH];
  const bool color = enable_stream_[COLOR];
  gpu_speckle_filter_active_ = gpu_speckle_filter_ && depth && enable_noise_removal_filter_;
  gpu_depth_alignment_active_ = gpu_depth_alignment_ && depth && color && depth_registration_ &&
                                align_mode_ == "SW" && align_target_stream_ == OB_STREAM_COLOR;
  const auto undistortion = enable_undistortion_.find(COLOR);
  gpu_color_active_ = gpu_color_ && color && undistortion != enable_undistortion_.end() &&
                      undistortion->second && !shouldUseHwD2CColorUndistortion();
}

void OBCameraNode::createGpuPipeline() {
  gpu_.reset();
  if (!gpu_speckle_filter_active_ && !gpu_depth_alignment_active_ && !gpu_color_active_) {
    return;
  }
  GpuImagePipeline::Options options;
  options.speckle_filter = gpu_speckle_filter_active_;
  options.depth_alignment = gpu_depth_alignment_active_;
  options.color = gpu_color_active_;
  options.compare = gpu_compare_;
  options.compare_every_n = gpu_compare_every_n_;
  options.jpeg_quality = color_jpeg_quality_;
  // The device keeps the filter's parameters while its own filter is off;
  // what it holds now is what the SDK would have filtered with.
  options.speckle.max_diff = noise_removal_filter_min_diff_;
  options.speckle.max_size = noise_removal_filter_max_size_;
  TRY_EXECUTE_BLOCK({
    if (device_->isPropertySupported(OB_PROP_DEPTH_MAX_DIFF_INT, OB_PERMISSION_READ)) {
      options.speckle.max_diff = device_->getIntProperty(OB_PROP_DEPTH_MAX_DIFF_INT);
    }
    if (device_->isPropertySupported(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT, OB_PERMISSION_READ)) {
      options.speckle.max_size = device_->getIntProperty(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT);
    }
  });
  if (options.speckle.max_diff < 0 || options.speckle.max_size < 0) {
    RCLCPP_WARN_STREAM(logger_, "No noise removal filter parameters known: the SDK keeps its "
                                "filter and the GPU pipeline leaves it alone");
    options.speckle_filter = false;
    gpu_speckle_filter_active_ = false;
  }
  gpu_ = std::make_unique<GpuImagePipeline>(node_, options);
}

bool OBCameraNode::applyNoiseRemovalToGpu(bool enabled, int min_diff, int max_size) {
  if (!gpu_ || !gpu_speckle_filter_active_) {
    return false;
  }
  if (device_->isPropertySupported(OB_PROP_DEPTH_SOFT_FILTER_BOOL, OB_PERMISSION_READ_WRITE)) {
    device_->setBoolProperty(OB_PROP_DEPTH_SOFT_FILTER_BOOL, false);
  }
  gpu_->setSpeckle(enabled, min_diff, max_size);
  RCLCPP_INFO_STREAM(logger_, "GPU noise removal filter " << (enabled ? "on" : "off")
                                                         << ", min_diff " << min_diff
                                                         << ", max_size " << max_size);
  return true;
}

// The raw image's own readers; compressed ones are asked separately.
bool OBCameraNode::imageWanted(const stream_index_pair &stream_index) {
  if (save_images_[stream_index]) {
    return true;
  }
  const bool has_explicit_compressed_publisher =
      compressed_image_publishers_.count(stream_index) > 0 &&
      compressed_image_publishers_.at(stream_index);
  if (!use_intra_process_ && !has_explicit_compressed_publisher) {
    return node_->count_subscribers(stream_name_[stream_index] + "/image_raw") > 0;
  }
  return image_publishers_.count(stream_index) && image_publishers_.at(stream_index) &&
         image_publishers_.at(stream_index)->get_subscription_count() > 0;
}

bool OBCameraNode::depthDataWanted() {
  return imageWanted(DEPTH) || hasCompressedImageSubscriber(DEPTH) || enable_d2c_viewer_ ||
         (depth_cloud_pub_ && depth_cloud_pub_->get_subscription_count() > 0) ||
         (depth_registration_cloud_pub_ &&
          depth_registration_cloud_pub_->get_subscription_count() > 0) ||
         (depth_unaligned_publisher_ && depth_unaligned_publisher_->get_subscription_count() > 0);
}

void OBCameraNode::processGpuColorFrameSet(const std::shared_ptr<ob::FrameSet> &frame_set) {
  auto color_frame = frame_set->colorFrame();
  if (!color_frame) {
    return;
  }
  if (color_frame->getFormat() != OB_FORMAT_MJPG) {
    RCLCPP_ERROR_STREAM_THROTTLE(logger_, *node_->get_clock(), 10000,
                                 "The GPU colour path needs MJPG colour, the camera sends format "
                                     << color_frame->getFormat());
    return;
  }
  const bool colored_cloud = enable_colored_point_cloud_ && depth_registration_cloud_pub_ &&
                             depth_registration_cloud_pub_->get_subscription_count() > 0;
  const bool jpeg_wanted = hasCompressedImageSubscriber(COLOR);
  const bool raw_wanted = colored_cloud || imageWanted(COLOR);
  std::shared_ptr<ob::Frame> rectified;
  std::vector<uint8_t> jpeg;
  if (!gpu_->processColor(color_frame, raw_wanted, jpeg_wanted, &rectified, &jpeg)) {
    // The failure is logged; the SDK's undistortion still gives the frame.
    rectified = gpu_->sdkRectify(color_frame);
    if (!rectified) {
      return;
    }
    jpeg.clear();
  }
  frame_set->pushFrame(rectified);
  is_color_frame_decoded_ = true;
  if (jpeg_wanted && !jpeg.empty()) {
    const auto timestamp = fromUsToROSTime(getFrameTimestampUs(rectified));
    publishJpegColorImage(std::move(jpeg), COLOR, timestamp, optical_frame_id_[COLOR]);
  }
  onNewFrameCallback(rectified, COLOR);
}

void OBCameraNode::publishJpegColorImage(std::vector<uint8_t> jpeg,
                                         const stream_index_pair &stream_index,
                                         const rclcpp::Time &timestamp,
                                         const std::string &frame_id) {
  auto it = compressed_image_publishers_.find(stream_index);
  if (it == compressed_image_publishers_.end() || !it->second) {
    return;
  }
  sensor_msgs::msg::CompressedImage msg;
  msg.header.stamp = timestamp;
  msg.header.frame_id = frame_id;
  // compressed_image_transport's spelling for an rgb8 image, which Howler
  // and image_transport tools decode.
  msg.format = "rgb8; jpeg compressed bgr8";
  msg.data = std::move(jpeg);
  recordStreamStatus(stream_name_[stream_index] + "/image_raw/compressed", msg.header.stamp);
  it->second->publish(std::move(msg));
}

}  // namespace orbbec_camera
