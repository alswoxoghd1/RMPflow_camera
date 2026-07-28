#include "rb10_rmpflow_rviz/casadi_rb10_kinematics.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <tinyxml2.h>

namespace rb10_rmpflow_rviz
{

namespace
{

struct UrdfJoint
{
  std::string name;
  std::string type;
  std::string parent;
  std::string child;
  Eigen::Vector3d xyz{Eigen::Vector3d::Zero()};
  Eigen::Vector3d rpy{Eigen::Vector3d::Zero()};
  Eigen::Vector3d axis{Eigen::Vector3d::UnitZ()};
  int q_index{-1};
};

Eigen::Vector3d parse_vector3(const char * text, const Eigen::Vector3d & fallback)
{
  if (text == nullptr) {
    return fallback;
  }
  std::istringstream stream(text);
  Eigen::Vector3d result = fallback;
  if (!(stream >> result.x() >> result.y() >> result.z())) {
    throw std::runtime_error(std::string("Invalid URDF vector: ") + text);
  }
  return result;
}

std::string required_attribute(
  const tinyxml2::XMLElement * element,
  const char * attribute,
  const std::string & context)
{
  if (element == nullptr || element->Attribute(attribute) == nullptr) {
    throw std::runtime_error("Missing URDF attribute " + context + "." + attribute);
  }
  return element->Attribute(attribute);
}

casadi::DM eigen_to_dm(const Eigen::MatrixXd & value)
{
  casadi::DM out = casadi::DM::zeros(value.rows(), value.cols());
  for (Eigen::Index col = 0; col < value.cols(); ++col) {
    for (Eigen::Index row = 0; row < value.rows(); ++row) {
      out(row, col) = value(row, col);
    }
  }
  return out;
}

Eigen::Matrix3d rpy_rotation(const Eigen::Vector3d & rpy)
{
  return (
    Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX())).toRotationMatrix();
}

casadi::MX homogeneous(
  const casadi::MX & rotation,
  const casadi::MX & translation)
{
  return casadi::MX::vertcat({
    casadi::MX::horzcat({rotation, translation}),
    casadi::DM(std::vector<double>{0.0, 0.0, 0.0, 1.0}).T()
  });
}

casadi::MX origin_transform(const UrdfJoint & joint)
{
  const casadi::DM rotation = eigen_to_dm(rpy_rotation(joint.rpy));
  const casadi::DM translation = eigen_to_dm(joint.xyz);
  return homogeneous(rotation, translation);
}

casadi::MX revolute_rotation(const Eigen::Vector3d & raw_axis, const casadi::MX & angle)
{
  const double norm = raw_axis.norm();
  if (norm <= 1e-12) {
    throw std::runtime_error("URDF revolute joint has a zero axis");
  }
  const Eigen::Vector3d axis = raw_axis / norm;
  Eigen::Matrix3d skew;
  skew <<
    0.0, -axis.z(), axis.y(),
    axis.z(), 0.0, -axis.x(),
    -axis.y(), axis.x(), 0.0;
  const casadi::DM k = eigen_to_dm(skew);
  return
    casadi::DM::eye(3) +
    sin(angle) * k +
    (1.0 - cos(angle)) * casadi::MX::mtimes(k, k);
}

casadi::MX joint_motion_transform(const UrdfJoint & joint, const casadi::MX & q)
{
  if (joint.type == "fixed") {
    return casadi::DM::eye(4);
  }
  if (joint.type != "revolute" && joint.type != "continuous") {
    throw std::runtime_error(
            "CasADi RB10 kinematics supports fixed/revolute joints, got: " + joint.type);
  }
  if (joint.q_index < 0 || joint.q_index >= q.size1()) {
    throw std::runtime_error("Missing RB10 configuration index for joint " + joint.name);
  }
  return homogeneous(
    revolute_rotation(joint.axis, q(joint.q_index)),
    casadi::DM::zeros(3, 1));
}

casadi::MX stack_columns(const std::vector<casadi::MX> & values)
{
  if (values.empty()) {
    return casadi::MX::zeros(0, 1);
  }
  return casadi::MX::vertcat(values);
}

casadi::DM joint_vector_to_dm(const RB10Model::JointVector & value)
{
  casadi::DM out = casadi::DM::zeros(6, 1);
  for (int index = 0; index < 6; ++index) {
    out(index) = value[index];
  }
  return out;
}

Eigen::VectorXd dm_to_vector(const casadi::DM & value)
{
  const casadi::DM dense = casadi::DM::densify(value);
  const auto & data = dense.nonzeros();
  Eigen::VectorXd result(static_cast<Eigen::Index>(data.size()));
  for (std::size_t index = 0; index < data.size(); ++index) {
    result[static_cast<Eigen::Index>(index)] = data[index];
  }
  return result;
}

Eigen::MatrixXd dm_to_matrix(const casadi::DM & value)
{
  const casadi::DM dense = casadi::DM::densify(value);
  Eigen::MatrixXd result(dense.size1(), dense.size2());
  for (casadi_int col = 0; col < dense.size2(); ++col) {
    for (casadi_int row = 0; row < dense.size1(); ++row) {
      result(row, col) = static_cast<double>(dense(row, col));
    }
  }
  return result;
}

Eigen::Matrix3d unstack_rotation(const Eigen::VectorXd & stacked, std::size_t index)
{
  Eigen::Matrix3d result;
  const Eigen::Index offset = static_cast<Eigen::Index>(9 * index);
  for (int col = 0; col < 3; ++col) {
    for (int row = 0; row < 3; ++row) {
      result(row, col) = stacked[offset + 3 * col + row];
    }
  }
  return result;
}

}  // namespace

struct CasadiRb10Kinematics::Impl
{
  explicit Impl(const std::string & urdf_path)
  {
    tinyxml2::XMLDocument document;
    const auto status = document.LoadFile(urdf_path.c_str());
    if (status != tinyxml2::XML_SUCCESS) {
      throw std::runtime_error(
              "Unable to load RB10 URDF for CasADi kinematics: " + urdf_path);
    }
    const auto * robot = document.FirstChildElement("robot");
    if (robot == nullptr) {
      throw std::runtime_error("URDF has no <robot> root: " + urdf_path);
    }

    std::unordered_map<std::string, int> joint_indices;
    for (std::size_t index = 0; index < RB10Model::joint_names.size(); ++index) {
      joint_indices.emplace(RB10Model::joint_names[index], static_cast<int>(index));
    }

    for (auto * element = robot->FirstChildElement("joint"); element != nullptr;
      element = element->NextSiblingElement("joint"))
    {
      UrdfJoint joint;
      joint.name = required_attribute(element, "name", "joint");
      joint.type = required_attribute(element, "type", "joint " + joint.name);
      const auto * parent = element->FirstChildElement("parent");
      const auto * child = element->FirstChildElement("child");
      joint.parent = required_attribute(parent, "link", "joint " + joint.name + ".parent");
      joint.child = required_attribute(child, "link", "joint " + joint.name + ".child");
      if (const auto * origin = element->FirstChildElement("origin")) {
        joint.xyz = parse_vector3(origin->Attribute("xyz"), Eigen::Vector3d::Zero());
        joint.rpy = parse_vector3(origin->Attribute("rpy"), Eigen::Vector3d::Zero());
      }
      if (const auto * axis = element->FirstChildElement("axis")) {
        joint.axis = parse_vector3(axis->Attribute("xyz"), Eigen::Vector3d::UnitZ());
      }
      const auto q_index = joint_indices.find(joint.name);
      if (q_index != joint_indices.end()) {
        joint.q_index = q_index->second;
      }
      joints_by_child.emplace(joint.child, std::move(joint));
    }

    for (const auto * name : RB10Model::joint_names) {
      bool found = false;
      for (const auto & entry : joints_by_child) {
        found = found || entry.second.name == name;
      }
      if (!found) {
        throw std::runtime_error("RB10 actuated joint is absent from URDF: " + std::string(name));
      }
    }

    const casadi::MX q = casadi::MX::sym("q", 6, 1);
    std::unordered_map<std::string, casadi::MX> transform_cache;
    transform_cache.emplace("base_link", casadi::DM::eye(4));
    std::function<casadi::MX(const std::string &)> frame_transform;
    frame_transform = [this, &q, &transform_cache, &frame_transform](const std::string & frame) {
        const auto cached = transform_cache.find(frame);
        if (cached != transform_cache.end()) {
          return cached->second;
        }
        const auto joint_it = joints_by_child.find(frame);
        if (joint_it == joints_by_child.end()) {
          throw std::runtime_error("URDF frame is not connected to base_link: " + frame);
        }
        const auto & joint = joint_it->second;
        const casadi::MX transform = casadi::MX::mtimes({
          frame_transform(joint.parent),
          origin_transform(joint),
          joint_motion_transform(joint, q)
        });
        transform_cache.emplace(frame, transform);
        return transform;
      };

    std::vector<casadi::MX> link_positions;
    std::vector<casadi::MX> link_rotations;
    link_positions.reserve(RB10Model::LINK_COUNT);
    link_rotations.reserve(RB10Model::LINK_COUNT);
    for (const auto * frame : RB10Model::link_names) {
      const casadi::MX transform = frame_transform(frame);
      link_positions.push_back(transform(casadi::Slice(0, 3), 3));
      link_rotations.push_back(casadi::MX::reshape(
          transform(casadi::Slice(0, 3), casadi::Slice(0, 3)), 9, 1));
    }

    std::vector<casadi::MX> sensor_positions;
    sensor_positions.reserve(RB10Model::sensor_control_points.size());
    for (const auto & sensor : RB10Model::sensor_control_points) {
      const casadi::MX transform = frame_transform(sensor.frame_name);
      sensor_positions.push_back(transform(casadi::Slice(0, 3), 3));
    }

    const casadi::MX stacked_links = stack_columns(link_positions);
    const casadi::MX stacked_rotations = stack_columns(link_rotations);
    const casadi::MX stacked_sensors = stack_columns(sensor_positions);
    task_map = casadi::Function(
      "rb10_casadi_task_map",
      {q},
      {stacked_links, stacked_rotations, stacked_sensors},
      {"q"},
      {"link_positions", "link_rotations", "sensor_positions"});

    const casadi::MX qd = casadi::MX::sym("qd", 6, 1);
    const casadi::MX link_jacobian = casadi::MX::jacobian(stacked_links, q);
    const casadi::MX link_velocity = jtimes(stacked_links, q, qd);
    const casadi::MX link_curvature = jtimes(link_velocity, q, qd);
    const casadi::MX sensor_jacobian = casadi::MX::jacobian(stacked_sensors, q);
    const casadi::MX sensor_velocity = jtimes(stacked_sensors, q, qd);
    const casadi::MX sensor_curvature = jtimes(sensor_velocity, q, qd);

    std::vector<casadi::MX> angular_velocities;
    std::vector<casadi::MX> angular_curvatures;
    std::vector<casadi::MX> angular_jacobians;
    angular_velocities.reserve(RB10Model::LINK_COUNT);
    angular_curvatures.reserve(RB10Model::LINK_COUNT);
    angular_jacobians.reserve(RB10Model::LINK_COUNT);
    for (std::size_t index = 0; index < RB10Model::LINK_COUNT; ++index) {
      const casadi::MX rotation = casadi::MX::reshape(
        stacked_rotations(casadi::Slice(
          static_cast<casadi_int>(9 * index),
          static_cast<casadi_int>(9 * (index + 1)))), 3, 3);
      const casadi::MX rotation_velocity = jtimes(rotation, q, qd);
      const casadi::MX omega_skew = casadi::MX::mtimes(rotation_velocity, rotation.T());
      const casadi::MX omega = casadi::MX::vertcat({
        omega_skew(2, 1), omega_skew(0, 2), omega_skew(1, 0)
      });
      angular_velocities.push_back(omega);
      angular_curvatures.push_back(jtimes(omega, q, qd));
      angular_jacobians.push_back(casadi::MX::jacobian(omega, qd));
    }

    evaluation = casadi::Function(
      "rb10_casadi_kinematics",
      {q, qd},
      {
        stacked_links,
        stacked_rotations,
        link_jacobian,
        link_velocity,
        link_curvature,
        stack_columns(angular_jacobians),
        stack_columns(angular_velocities),
        stack_columns(angular_curvatures),
        stacked_sensors,
        sensor_jacobian,
        sensor_velocity,
        sensor_curvature
      });
  }

  std::unordered_map<std::string, UrdfJoint> joints_by_child;
  casadi::Function task_map;
  casadi::Function evaluation;
};

CasadiRb10Kinematics::CasadiRb10Kinematics(const std::string & urdf_path)
: impl_(std::make_shared<Impl>(urdf_path))
{}

const casadi::Function & CasadiRb10Kinematics::task_map_function() const
{
  return impl_->task_map;
}

KinematicsContext CasadiRb10Kinematics::forward_context(
  const JointVector & q,
  const JointVector & qd) const
{
  const std::vector<casadi::DM> inputs{joint_vector_to_dm(q), joint_vector_to_dm(qd)};
  const std::vector<casadi::DM> values = impl_->evaluation(inputs);
  if (values.size() != 12) {
    throw std::runtime_error("Unexpected CasADi RB10 kinematics output count");
  }

  const Eigen::VectorXd link_positions = dm_to_vector(values[0]);
  const Eigen::VectorXd link_rotations = dm_to_vector(values[1]);
  const Eigen::MatrixXd link_jacobians = dm_to_matrix(values[2]);
  const Eigen::VectorXd link_velocities = dm_to_vector(values[3]);
  const Eigen::VectorXd link_curvatures = dm_to_vector(values[4]);
  const Eigen::MatrixXd link_angular_jacobians = dm_to_matrix(values[5]);
  const Eigen::VectorXd link_angular_velocities = dm_to_vector(values[6]);
  const Eigen::VectorXd link_angular_curvatures = dm_to_vector(values[7]);
  const Eigen::VectorXd control_points = dm_to_vector(values[8]);
  const Eigen::MatrixXd control_point_jacobians = dm_to_matrix(values[9]);
  const Eigen::VectorXd control_point_velocities = dm_to_vector(values[10]);
  const Eigen::VectorXd control_point_curvatures = dm_to_vector(values[11]);

  KinematicsContext context;
  for (std::size_t index = 0; index < RB10Model::LINK_COUNT; ++index) {
    const Eigen::Index offset = static_cast<Eigen::Index>(3 * index);
    context.link_positions[index] = link_positions.segment<3>(offset);
    context.link_rotations[index] = unstack_rotation(link_rotations, index);
    context.link_jacobians[index] = link_jacobians.block<3, 6>(offset, 0);
    context.link_velocities[index] = link_velocities.segment<3>(offset);
    context.link_curvatures[index] = link_curvatures.segment<3>(offset);
    context.link_angular_velocities[index] =
      link_angular_velocities.segment<3>(offset);
    context.link_angular_curvatures[index] =
      link_angular_curvatures.segment<3>(offset);
    context.link_angular_jacobians[index] =
      link_angular_jacobians.block<3, 6>(offset, 0);
  }

  const std::size_t point_count = RB10Model::sensor_control_points.size();
  context.control_points.reserve(point_count);
  context.control_point_jacobians.reserve(point_count);
  context.control_point_velocities.reserve(point_count);
  context.control_point_curvatures.reserve(point_count);
  for (std::size_t index = 0; index < point_count; ++index) {
    const Eigen::Index offset = static_cast<Eigen::Index>(3 * index);
    context.control_points.push_back(ControlPoint{
      control_points.segment<3>(offset),
      RB10Model::sensor_control_points[index].radius
    });
    context.control_point_jacobians.push_back(
      control_point_jacobians.block<3, 6>(offset, 0));
    context.control_point_velocities.push_back(
      control_point_velocities.segment<3>(offset));
    context.control_point_curvatures.push_back(
      control_point_curvatures.segment<3>(offset));
  }

  context.tcp_position = context.link_positions[RB10Model::TCP_RMP];
  context.tcp_jacobian = context.link_jacobians[RB10Model::TCP_RMP];
  context.tcp_velocity = context.link_velocities[RB10Model::TCP_RMP];
  context.tcp_curvature = context.link_curvatures[RB10Model::TCP_RMP];
  return context;
}

}  // namespace rb10_rmpflow_rviz
