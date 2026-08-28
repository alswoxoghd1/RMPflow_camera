#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>

#include <Eigen/Geometry>

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/u_int8.hpp"

namespace rb10_rmpflow_rviz
{

namespace
{

class RmpflowBridgeNode : public rclcpp::Node
{
public:
  RmpflowBridgeNode()
  : Node("rmpflow_bridge")
  {
    declare_parameter("flag_topic", std::string("/RMP_flag"));
    declare_parameter("goal_topic", std::string("/RMP_goal"));
    declare_parameter("controller_goal_topic", std::string("/goal_pose"));
    declare_parameter("controller_command_topic", std::string("/position_controllers/commands"));
    declare_parameter("target_q_topic", std::string("/target_q"));
    declare_parameter("forward_target_q", true);
    declare_parameter("goal_frame_id", std::string("base_link"));
    declare_parameter("active_flag_value", 1);
    declare_parameter("command_forward_delay_ms", 100);
    declare_parameter("goal_filter_enabled", false);
    declare_parameter("goal_filter_rate_hz", 200.0);
    declare_parameter("goal_filter_time_constant_s", 0.08);
    declare_parameter("goal_filter_max_linear_speed_m_s", 0.30);
    declare_parameter("goal_filter_max_angular_speed_rad_s", 1.0);

    const auto flag_topic = get_parameter("flag_topic").as_string();
    const auto goal_topic = get_parameter("goal_topic").as_string();
    const auto controller_goal_topic = get_parameter("controller_goal_topic").as_string();
    const auto controller_command_topic = get_parameter("controller_command_topic").as_string();
    const auto target_q_topic = get_parameter("target_q_topic").as_string();
    forward_target_q_ = get_parameter("forward_target_q").as_bool();
    goal_frame_id_ = get_parameter("goal_frame_id").as_string();
    active_flag_value_ = static_cast<int>(get_parameter("active_flag_value").as_int());
    command_forward_delay_ = std::chrono::milliseconds(static_cast<int>(
      std::max<int64_t>(0, get_parameter("command_forward_delay_ms").as_int())));
    goal_filter_enabled_ = get_parameter("goal_filter_enabled").as_bool();
    goal_filter_rate_hz_ = std::max(get_parameter("goal_filter_rate_hz").as_double(), 1.0);
    goal_filter_time_constant_s_ =
      std::max(get_parameter("goal_filter_time_constant_s").as_double(), 0.0);
    goal_filter_max_linear_speed_m_s_ =
      std::max(get_parameter("goal_filter_max_linear_speed_m_s").as_double(), 0.0);
    goal_filter_max_angular_speed_rad_s_ =
      std::max(get_parameter("goal_filter_max_angular_speed_rad_s").as_double(), 0.0);

    controller_goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      controller_goal_topic,
      10);
    if (forward_target_q_) {
      target_q_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(target_q_topic, 10);
    }

    flag_sub_ = create_subscription<std_msgs::msg::UInt8>(
      flag_topic,
      10,
      std::bind(&RmpflowBridgeNode::on_flag, this, std::placeholders::_1));
    goal_sub_ = create_subscription<geometry_msgs::msg::Pose>(
      goal_topic,
      10,
      std::bind(&RmpflowBridgeNode::on_goal, this, std::placeholders::_1));
    if (forward_target_q_) {
      controller_command_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        controller_command_topic,
        10,
        std::bind(&RmpflowBridgeNode::on_controller_command, this, std::placeholders::_1));
    }
    if (goal_filter_enabled_) {
      const auto filter_period = std::chrono::duration<double>(1.0 / goal_filter_rate_hz_);
      goal_filter_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(filter_period),
        std::bind(&RmpflowBridgeNode::on_goal_filter_timer, this));
      last_goal_filter_update_ = std::chrono::steady_clock::now();
    }

    RCLCPP_INFO(
      get_logger(),
      "rmpflow_bridge ready: %s + %s -> %s (%s)",
      flag_topic.c_str(),
      goal_topic.c_str(),
      target_q_topic.c_str(),
      forward_target_q_ ? "forwarding target_q" : "goal forwarding only");
    if (goal_filter_enabled_) {
      RCLCPP_INFO(
        get_logger(),
        "Goal filter enabled: %.1f Hz, tau=%.3f s, linear<=%.3f m/s, angular<=%.3f rad/s",
        goal_filter_rate_hz_,
        goal_filter_time_constant_s_,
        goal_filter_max_linear_speed_m_s_,
        goal_filter_max_angular_speed_rad_s_);
    }
  }

private:
  void on_flag(const std_msgs::msg::UInt8::SharedPtr msg)
  {
    std::scoped_lock lock(mutex_);

    const int flag_value = static_cast<int>(msg->data);
    if (!last_flag_value_seen_ || flag_value != last_flag_value_) {
      if (flag_value == active_flag_value_) {
        RCLCPP_INFO(get_logger(), "RMP Init!!!!!");
      } else if (flag_value == 0) {
        RCLCPP_INFO(get_logger(), "RMP Standby!!!!");
      }
      last_flag_value_seen_ = true;
      last_flag_value_ = flag_value;
    }

    const bool requested_active = flag_value == active_flag_value_;
    if (requested_active == active_) {
      return;
    }
    active_ = requested_active;

    if (!active_) {
      command_forwarding_enabled_ = false;
      last_goal_filter_update_ = std::chrono::steady_clock::now();
      RCLCPP_INFO(
        get_logger(),
        forward_target_q_ ? "RMP bridge deactivated; target_q forwarding paused" :
        "RMP bridge deactivated; goal forwarding paused");
      return;
    }

    if (!goal_received_) {
      command_forwarding_enabled_ = false;
      RCLCPP_WARN(
        get_logger(),
        forward_target_q_ ?
        "RMP bridge activated but no /RMP_goal has been received yet; waiting before forwarding target_q" :
        "RMP bridge activated but no /RMP_goal has been received yet; waiting before forwarding goal_pose");
      return;
    }

    if (!goal_filter_enabled_ || !filtered_goal_initialized_) {
      filtered_goal_ = latest_goal_;
      filtered_goal_initialized_ = true;
    }
    publish_external_goal_locked(goal_filter_enabled_ ? filtered_goal_ : latest_goal_);
    enable_command_forwarding_after_delay_locked();
    RCLCPP_INFO(
      get_logger(),
      forward_target_q_ ?
      "RMP bridge activated; forwarding target_q once the new goal settles" :
      "RMP bridge activated; forwarding /RMP_goal to /goal_pose");
  }

  void on_goal(const geometry_msgs::msg::Pose::SharedPtr msg)
  {
    std::scoped_lock lock(mutex_);

    geometry_msgs::msg::Pose next_goal = *msg;
    if (!sanitize_goal(next_goal)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "Rejected non-finite /RMP_goal");
      return;
    }
    latest_goal_ = next_goal;
    goal_received_ = true;
    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      1000,
      "Received /RMP_goal: position=(%.4f, %.4f, %.4f), orientation=(%.4f, %.4f, %.4f, %.4f)",
      latest_goal_.position.x,
      latest_goal_.position.y,
      latest_goal_.position.z,
      latest_goal_.orientation.x,
      latest_goal_.orientation.y,
      latest_goal_.orientation.z,
      latest_goal_.orientation.w);
    if (!active_) {
      return;
    }

    if (goal_filter_enabled_) {
      if (!filtered_goal_initialized_) {
        filtered_goal_ = latest_goal_;
        filtered_goal_initialized_ = true;
        publish_external_goal_locked(filtered_goal_);
        enable_command_forwarding_after_delay_locked();
      }
      return;
    }

    publish_external_goal_locked(latest_goal_);
    if (!command_forwarding_enabled_) {
      enable_command_forwarding_after_delay_locked();
    }
  }

  void on_goal_filter_timer()
  {
    std::scoped_lock lock(mutex_);
    const auto now_steady = std::chrono::steady_clock::now();
    const double nominal_dt = 1.0 / goal_filter_rate_hz_;
    double dt = std::chrono::duration<double>(now_steady - last_goal_filter_update_).count();
    last_goal_filter_update_ = now_steady;
    // A delayed timer callback must not turn into one large allowed step.
    dt = std::clamp(dt, 0.25 * nominal_dt, 2.0 * nominal_dt);

    if (!active_ || !goal_received_) {
      return;
    }
    if (!filtered_goal_initialized_) {
      filtered_goal_ = latest_goal_;
      filtered_goal_initialized_ = true;
    } else {
      update_filtered_goal_locked(dt);
    }
    publish_external_goal_locked(filtered_goal_);
    if (!command_forwarding_enabled_) {
      enable_command_forwarding_after_delay_locked();
    }
  }
  void on_controller_command(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    std::scoped_lock lock(mutex_);

    if (!forward_target_q_ || !target_q_pub_) {
      return;
    }
    if (!command_forwarding_enabled_) {
      return;
    }
    if (std::chrono::steady_clock::now() < command_forwarding_allowed_after_) {
      return;
    }
    target_q_pub_->publish(*msg);
  }

  static bool sanitize_goal(geometry_msgs::msg::Pose & goal)
  {
    if (
      !std::isfinite(goal.position.x) ||
      !std::isfinite(goal.position.y) ||
      !std::isfinite(goal.position.z))
    {
      return false;
    }
    Eigen::Quaterniond orientation(
      goal.orientation.w,
      goal.orientation.x,
      goal.orientation.y,
      goal.orientation.z);
    if (!orientation.coeffs().allFinite() || orientation.norm() < 1e-9) {
      orientation = Eigen::Quaterniond::Identity();
    } else {
      orientation.normalize();
    }
    goal.orientation.x = orientation.x();
    goal.orientation.y = orientation.y();
    goal.orientation.z = orientation.z();
    goal.orientation.w = orientation.w();
    return true;
  }

  static Eigen::Quaterniond goal_orientation(const geometry_msgs::msg::Pose & goal)
  {
    return Eigen::Quaterniond(
      goal.orientation.w,
      goal.orientation.x,
      goal.orientation.y,
      goal.orientation.z).normalized();
  }

  static void set_goal_orientation(
    geometry_msgs::msg::Pose & goal,
    const Eigen::Quaterniond & orientation)
  {
    const Eigen::Quaterniond normalized = orientation.normalized();
    goal.orientation.x = normalized.x();
    goal.orientation.y = normalized.y();
    goal.orientation.z = normalized.z();
    goal.orientation.w = normalized.w();
  }

  void update_filtered_goal_locked(double dt)
  {
    const Eigen::Vector3d filtered_position(
      filtered_goal_.position.x,
      filtered_goal_.position.y,
      filtered_goal_.position.z);
    const Eigen::Vector3d raw_position(
      latest_goal_.position.x,
      latest_goal_.position.y,
      latest_goal_.position.z);
    const double alpha = goal_filter_time_constant_s_ > 0.0 ?
      1.0 - std::exp(-dt / goal_filter_time_constant_s_) :
      1.0;
    Eigen::Vector3d position_step = alpha * (raw_position - filtered_position);
    const double max_position_step = goal_filter_max_linear_speed_m_s_ * dt;
    if (max_position_step > 0.0 && position_step.norm() > max_position_step) {
      position_step *= max_position_step / position_step.norm();
    }
    const Eigen::Vector3d next_position = filtered_position + position_step;
    filtered_goal_.position.x = next_position.x();
    filtered_goal_.position.y = next_position.y();
    filtered_goal_.position.z = next_position.z();

    Eigen::Quaterniond filtered_orientation = goal_orientation(filtered_goal_);
    Eigen::Quaterniond raw_orientation = goal_orientation(latest_goal_);
    if (filtered_orientation.dot(raw_orientation) < 0.0) {
      raw_orientation.coeffs() *= -1.0;
    }
    const double dot = std::clamp(filtered_orientation.dot(raw_orientation), -1.0, 1.0);
    const double angle = 2.0 * std::acos(std::abs(dot));
    double orientation_fraction = alpha;
    const double max_angle_step = goal_filter_max_angular_speed_rad_s_ * dt;
    if (angle > 1e-12 && max_angle_step > 0.0) {
      orientation_fraction = std::min(orientation_fraction, max_angle_step / angle);
    }
    set_goal_orientation(
      filtered_goal_,
      filtered_orientation.slerp(std::clamp(orientation_fraction, 0.0, 1.0), raw_orientation));
  }

  void publish_external_goal_locked(const geometry_msgs::msg::Pose & goal)
  {
    geometry_msgs::msg::PoseStamped goal_msg;
    goal_msg.header.stamp = now();
    goal_msg.header.frame_id = goal_frame_id_;
    goal_msg.pose = goal;
    controller_goal_pub_->publish(goal_msg);
  }

  void enable_command_forwarding_after_delay_locked()
  {
    command_forwarding_enabled_ = true;
    command_forwarding_allowed_after_ = std::chrono::steady_clock::now() + command_forward_delay_;
  }

  std::mutex mutex_;
  std::string goal_frame_id_{"base_link"};
  bool forward_target_q_{true};
  int active_flag_value_{1};
  std::chrono::milliseconds command_forward_delay_{100};
  std::chrono::steady_clock::time_point command_forwarding_allowed_after_{};

  bool active_{false};
  bool goal_received_{false};
  bool command_forwarding_enabled_{false};
  bool last_flag_value_seen_{false};
  int last_flag_value_{0};
  bool goal_filter_enabled_{false};
  bool filtered_goal_initialized_{false};
  double goal_filter_rate_hz_{200.0};
  double goal_filter_time_constant_s_{0.08};
  double goal_filter_max_linear_speed_m_s_{0.30};
  double goal_filter_max_angular_speed_rad_s_{1.0};
  std::chrono::steady_clock::time_point last_goal_filter_update_{};

  geometry_msgs::msg::Pose latest_goal_;
  geometry_msgs::msg::Pose filtered_goal_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr controller_goal_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_q_pub_;

  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr flag_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr goal_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr controller_command_sub_;
  rclcpp::TimerBase::SharedPtr goal_filter_timer_;
};

}  // namespace

}  // namespace rb10_rmpflow_rviz

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rb10_rmpflow_rviz::RmpflowBridgeNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
