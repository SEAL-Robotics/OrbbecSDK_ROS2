// Copyright 2026 SEAL Robotics
//
// What the driver relies on when components share its container with
// intra-process delivery on: its configured QoS works unchanged, a depth
// image reaches a sensor-data subscriber in the same process without a
// copy, and a transient-local message still reaches a late subscriber.

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

#include <chrono>
#include <memory>

namespace {

class IntraProcess : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  std::shared_ptr<rclcpp::Node> node(const char *name) {
    return std::make_shared<rclcpp::Node>(name,
                                          rclcpp::NodeOptions().use_intra_process_comms(true));
  }

  void spinUntil(rclcpp::executors::SingleThreadedExecutor &executor,
                 const std::function<bool()> &done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!done() && std::chrono::steady_clock::now() < deadline) {
      executor.spin_some(std::chrono::milliseconds(20));
    }
  }
};

TEST_F(IntraProcess, ADepthImageReachesTheGateWithoutACopy) {
  auto driver = node("driver");
  auto gate = node("gate");
  // The driver's default image QoS, and the depth gate's subscription.
  auto publisher = driver->create_publisher<sensor_msgs::msg::Image>(
      "depth/image_raw", rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default),
                                     rmw_qos_profile_default));
  const void *received = nullptr;
  auto subscription = gate->create_subscription<sensor_msgs::msg::Image>(
      "depth/image_raw", rclcpp::SensorDataQoS(),
      [&](sensor_msgs::msg::Image::ConstSharedPtr msg) { received = msg->data.data(); });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(driver);
  executor.add_node(gate);
  spinUntil(executor, [&] { return publisher->get_subscription_count() > 0; });

  auto image = std::make_unique<sensor_msgs::msg::Image>();
  image->data.resize(1280 * 800 * 2);
  const void *sent = image->data.data();
  publisher->publish(std::move(image));
  spinUntil(executor, [&] { return received != nullptr; });
  EXPECT_EQ(received, sent);
}

TEST_F(IntraProcess, TransientLocalStillReachesALateSubscriber) {
  auto driver = node("status_driver");
  // The driver's extrinsics and device status QoS.
  auto publisher = driver->create_publisher<std_msgs::msg::String>(
      "depth_to_color", rclcpp::QoS(1).transient_local());
  std_msgs::msg::String message;
  message.data = "latched";
  publisher->publish(message);

  auto late = node("late_reader");
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

}  // namespace
