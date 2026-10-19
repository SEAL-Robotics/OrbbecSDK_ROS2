// Copyright 2026 SEAL Robotics
//
// What the driver relies on when components share its container with
// intra-process delivery on: its QoS works intra-process (only a history
// rclcpp refuses is changed), a depth image published through
// image_transport reaches an in-container subscriber without a copy while
// image_transport's plugin topics stay, a frame read only in-process is not
// published to DDS at all, transient-local still reaches a late subscriber,
// and the driver node itself runs in a multi-threaded container.

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <thread>

#include "orbbec_camera/image_publisher.h"
#include "orbbec_camera/ob_camera_node_driver.h"
#include "orbbec_camera/utils.h"

namespace orbbec_camera {
namespace {

std::shared_ptr<rclcpp::Node> makeNode(const std::string &name, bool intra_process = true) {
  return std::make_shared<rclcpp::Node>(name,
                                        rclcpp::NodeOptions().use_intra_process_comms(intra_process));
}

void spinUntil(rclcpp::Executor &executor, const std::function<bool()> &done,
               std::chrono::milliseconds limit = std::chrono::milliseconds(5000)) {
  const auto deadline = std::chrono::steady_clock::now() + limit;
  while (!done() && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(std::chrono::milliseconds(20));
  }
}

class IntraProcess : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
};

TEST(IntraProcessQos, OnlyAHistoryRclcppRefusesChanges) {
  rmw_qos_profile_t keep_all = rmw_qos_profile_default;
  keep_all.history = RMW_QOS_POLICY_HISTORY_KEEP_ALL;
  rmw_qos_profile_t system_default = rmw_qos_profile_system_default;
  rmw_qos_profile_t no_depth = rmw_qos_profile_default;
  no_depth.depth = 0;
  for (const auto &qos : {keep_all, system_default, no_depth}) {
    const auto out = intraProcessQos(qos);
    EXPECT_EQ(out.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
    EXPECT_EQ(out.depth, std::max<size_t>(qos.depth, 10));
    EXPECT_EQ(out.reliability, qos.reliability);
    EXPECT_EQ(out.durability, qos.durability);
  }
  rmw_qos_profile_t latched = rmw_qos_profile_default;
  latched.depth = 1;
  latched.durability = RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL;
  const auto out = intraProcessQos(latched);
  EXPECT_EQ(out.history, latched.history);
  EXPECT_EQ(out.depth, 1u);
  EXPECT_EQ(out.durability, latched.durability);
  const auto sensor = intraProcessQos(rmw_qos_profile_sensor_data);
  EXPECT_EQ(sensor.depth, rmw_qos_profile_sensor_data.depth);
  EXPECT_EQ(sensor.reliability, rmw_qos_profile_sensor_data.reliability);
}

TEST_F(IntraProcess, ImageTransportDepthReachesTheGateWithoutACopy) {
  auto driver = makeNode("driver");
  auto gate = makeNode("gate");
  auto publisher = getGlobalImageTransportPublisher(*driver, "depth/image_raw",
                                                    rmw_qos_profile_default);
  const void *received = nullptr;
  auto subscription = gate->create_subscription<sensor_msgs::msg::Image>(
      "depth/image_raw", rclcpp::SensorDataQoS(),
      [&](sensor_msgs::msg::Image::ConstSharedPtr msg) { received = msg->data.data(); });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(driver);
  executor.add_node(gate);
  spinUntil(executor, [&] { return publisher->get_subscription_count() > 0; });

  auto image = std::make_unique<sensor_msgs::msg::Image>();
  image->encoding = "16UC1";
  image->width = 1280;
  image->height = 800;
  image->step = 2560;
  image->data.resize(1280 * 800 * 2);
  const void *sent = image->data.data();
  publisher->publish(std::move(image));
  spinUntil(executor, [&] { return received != nullptr; });
  EXPECT_EQ(received, sent);
  releaseGlobalImageTransportPublisher(*driver, "depth/image_raw");
}

std::set<std::string> topicsOf(const std::shared_ptr<rclcpp::Node> &node, const std::string &base) {
  std::set<std::string> topics;
  for (const auto &entry : node->get_topic_names_and_types()) {
    if (entry.first.rfind(base, 0) == 0) {
      topics.insert(entry.first);
    }
  }
  return topics;
}

TEST_F(IntraProcess, EveryReaderGetsTheFrameWhenTheGateIsNotAlone) {
  // The frame moved into image_transport must still reach a second reader,
  // in-process or not: no reader may see a moved-from message.
  auto driver = makeNode("shared_driver");
  auto gate = makeNode("shared_gate");
  auto other = makeNode("shared_other", false);
  auto publisher = getGlobalImageTransportPublisher(*driver, "shared/depth/image_raw",
                                                    rmw_qos_profile_default);
  size_t gate_bytes = 0;
  size_t other_bytes = 0;
  auto gate_subscription = gate->create_subscription<sensor_msgs::msg::Image>(
      "shared/depth/image_raw", rclcpp::SensorDataQoS(),
      [&](sensor_msgs::msg::Image::ConstSharedPtr msg) { gate_bytes = msg->data.size(); });
  auto other_subscription = other->create_subscription<sensor_msgs::msg::Image>(
      "shared/depth/image_raw", rclcpp::SensorDataQoS(),
      [&](sensor_msgs::msg::Image::ConstSharedPtr msg) { other_bytes = msg->data.size(); });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(driver);
  executor.add_node(gate);
  executor.add_node(other);
  spinUntil(executor, [&] { return publisher->get_subscription_count() >= 2; });

  auto image = std::make_unique<sensor_msgs::msg::Image>();
  image->encoding = "16UC1";
  image->width = 64;
  image->height = 40;
  image->step = 128;
  image->data.assign(64 * 40 * 2, 7);
  publisher->publish(std::move(image));
  spinUntil(executor, [&] { return gate_bytes != 0 && other_bytes != 0; });
  EXPECT_EQ(gate_bytes, 64u * 40u * 2u);
  EXPECT_EQ(other_bytes, 64u * 40u * 2u);
  releaseGlobalImageTransportPublisher(*driver, "shared/depth/image_raw");
}

TEST_F(IntraProcess, ImageTransportPluginTopicsStay) {
  // Whatever plugins this host has (compressed, compressedDepth on the pod),
  // the intra-process driver offers the same topics as the DDS-only one.
  auto plain = makeNode("plain_driver", false);
  auto plain_publisher =
      getGlobalImageTransportPublisher(*plain, "/plain/depth/image_raw", rmw_qos_profile_default);
  auto shared = makeNode("ipc_driver", true);
  auto shared_publisher =
      getGlobalImageTransportPublisher(*shared, "/ipc/depth/image_raw", rmw_qos_profile_default);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  std::set<std::string> plain_suffixes, ipc_suffixes;
  for (const auto &topic : topicsOf(plain, "/plain/")) {
    plain_suffixes.insert(topic.substr(std::string("/plain").size()));
  }
  for (const auto &topic : topicsOf(shared, "/ipc/")) {
    ipc_suffixes.insert(topic.substr(std::string("/ipc").size()));
  }
  EXPECT_FALSE(plain_suffixes.empty());
  EXPECT_EQ(plain_suffixes, ipc_suffixes);
  releaseGlobalImageTransportPublisher(*plain, "/plain/depth/image_raw");
  releaseGlobalImageTransportPublisher(*shared, "/ipc/depth/image_raw");
}

TEST_F(IntraProcess, AFrameReadOnlyInProcessIsNotPublishedToDds) {
  auto driver = makeNode("only_ipc_driver");
  auto gate = makeNode("only_ipc_gate");
  auto publisher = driver->create_publisher<sensor_msgs::msg::Image>(
      "only_ipc/depth", rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default),
                                    rmw_qos_profile_default));
  auto subscription = gate->create_subscription<sensor_msgs::msg::Image>(
      "only_ipc/depth", rclcpp::SensorDataQoS(), [](sensor_msgs::msg::Image::ConstSharedPtr) {});
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(driver);
  executor.add_node(gate);
  spinUntil(executor, [&] { return publisher->get_subscription_count() > 0; });
  // rclcpp publishes to DDS only for subscriptions beyond the intra-process ones.
  EXPECT_EQ(publisher->get_subscription_count(), 1u);
  EXPECT_EQ(publisher->get_intra_process_subscription_count(), 1u);
}

TEST_F(IntraProcess, TransientLocalStillReachesALateSubscriber) {
  auto driver = makeNode("status_driver");
  auto publisher = driver->create_publisher<std_msgs::msg::String>(
      "depth_to_color", rclcpp::QoS(1).transient_local());
  std_msgs::msg::String message;
  message.data = "latched";
  publisher->publish(message);

  auto late = makeNode("late_reader");
  std::string received;
  auto subscription = late->create_subscription<std_msgs::msg::String>(
      "depth_to_color", rclcpp::QoS(1).transient_local(),
      [&](std_msgs::msg::String::ConstSharedPtr msg) { received = msg->data; });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(driver);
  executor.add_node(late);
  spinUntil(executor, [&] { return !received.empty(); });
  EXPECT_EQ(received, "latched");
}

TEST_F(IntraProcess, TheDriverRunsInAMultiThreadedContainer) {
  // The driver opens any camera it finds, USB included, so this runs only
  // where asked to, on a machine with none attached.
  const char *enabled = std::getenv("ORBBEC_TEST_DRIVER_WITHOUT_CAMERA");
  if (enabled == nullptr || std::string(enabled) != "1") {
    GTEST_SKIP() << "set ORBBEC_TEST_DRIVER_WITHOUT_CAMERA=1 on a machine with no Orbbec attached";
  }
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  options.append_parameter_override("enumerate_net_device", false);
  auto driver = std::make_shared<OBCameraNodeDriver>(options);
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(driver);
  std::thread spinner([&executor] { executor.spin(); });
  std::this_thread::sleep_for(std::chrono::seconds(2));
  executor.cancel();
  spinner.join();
  EXPECT_TRUE(driver->get_node_options().use_intra_process_comms());
  EXPECT_FALSE(driver->get_fully_qualified_name() == std::string());
  executor.remove_node(driver);
}

}  // namespace
}  // namespace orbbec_camera
