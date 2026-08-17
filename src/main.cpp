/*
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/io.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cstring>

#include "lidar_detect.hpp"
#include "qr_detect.hpp"
#include "data_preprocess.hpp"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("mono_qr_pattern");

  // 读取参数
  Params params = loadParameters(node);

  // 初始化 QR 检测和 LiDAR 检测
  QRDetectPtr qrDetectPtr;
  qrDetectPtr.reset(new QRDetect(node, params));

  LidarDetectPtr lidarDetectPtr;
  lidarDetectPtr.reset(new LidarDetect(node, params));

  DataPreprocessPtr dataPreprocessPtr;
  dataPreprocessPtr.reset(new DataPreprocess(params));

  // 读取图像和点云
  cv::Mat img_input = dataPreprocessPtr->img_input_;
  pcl::PointCloud<Common::Point>::Ptr cloud_input = dataPreprocessPtr->cloud_input_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*cloud_input, *cloud_xyz);

  // 检测 QR 码
  PointCloud<PointXYZ>::Ptr qr_center_cloud(new PointCloud<PointXYZ>);
  qr_center_cloud->reserve(4);
  qrDetectPtr->detect_qr(img_input, qr_center_cloud);

  // 检测 LiDAR 数据（与 ROS1 相同：按雷达类型分流）
  PointCloud<PointXYZ>::Ptr lidar_center_cloud(new PointCloud<PointXYZ>);
  lidar_center_cloud->reserve(4);
  switch (dataPreprocessPtr->lidar_type_)
  {
  case LiDARType::Solid:
    lidarDetectPtr->detect_solid_lidar(cloud_input, lidar_center_cloud);
    break;
  case LiDARType::Mech:
    lidarDetectPtr->detect_mech_lidar(cloud_input, lidar_center_cloud);
    break;
  default:
    RCLCPP_ERROR(node->get_logger(), "[Main] Unknown LiDAR type.");
    break;
  }

  // 对 QR 和 LiDAR 检测到的圆心进行排序（LiDAR 使用配置的轴映射）
  PointCloud<PointXYZ>::Ptr qr_centers(new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr lidar_centers(new PointCloud<PointXYZ>);
  sortPatternCenters(qr_center_cloud, qr_centers, "camera");
  sortPatternCenters(lidar_center_cloud, lidar_centers, "lidar", &params);

  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_lidar_centers(new pcl::PointCloud<pcl::PointXYZ>);
  double rmse = -1.0;
  const bool have_four_pairs = (qr_centers->size() == 4 && lidar_centers->size() == 4);
  if (!have_four_pairs)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "[Main] Skip SVD: qr_centers=%zu lidar_centers=%zu (need 4/4).",
                 qr_centers->size(), lidar_centers->size());
  }
  else
  {
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
    svd.estimateRigidTransformation(*lidar_centers, *qr_centers, transformation);
    aligned_lidar_centers->reserve(lidar_centers->size());
    alignPointCloud(lidar_centers, aligned_lidar_centers, transformation);
    rmse = computeRMSE(qr_centers, aligned_lidar_centers);
    if (rmse > 0)
      RCLCPP_INFO(node->get_logger(), "[Result] RMSE: %.4f m", rmse);
  }

  Eigen::Matrix4f external_transformation = Eigen::Matrix4f::Identity();
  bool external_loaded = false;
  const bool need_external_extrinsic =
      (params.use_external_extrinsic_eval || params.use_external_extrinsic_for_coloring) &&
      !params.external_calib_result_path.empty();
  if (need_external_extrinsic)
  {
    external_loaded = loadExtrinsicFromCalibResult(params.external_calib_result_path, external_transformation);
    if (!external_loaded)
    {
      RCLCPP_WARN(node->get_logger(), "[Warn] Failed to load external calib file: %s",
                  params.external_calib_result_path.c_str());
    }
  }

  double external_rmse_m = -1.0;
  if (params.use_external_extrinsic_eval)
  {
    if (external_loaded)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_lidar_centers_external(new pcl::PointCloud<pcl::PointXYZ>);
      aligned_lidar_centers_external->reserve(lidar_centers->size());
      alignPointCloud(lidar_centers, aligned_lidar_centers_external, external_transformation);
      external_rmse_m = computeRMSE(qr_centers, aligned_lidar_centers_external);
      if (external_rmse_m >= 0.0)
      {
        RCLCPP_INFO(node->get_logger(), "[Result] External extrinsic RMSE: %.4f m", external_rmse_m);
      }
    }
    else
    {
      RCLCPP_WARN(node->get_logger(), "[Warn] External RMSE evaluation skipped because external extrinsic is unavailable.");
    }
  }
  if (qrDetectPtr->reprojection_rmse_px_ >= 0.0)
  {
    RCLCPP_INFO(node->get_logger(), "[Result] Reprojection RMSE: %.4f px", qrDetectPtr->reprojection_rmse_px_);
  }

  RCLCPP_INFO(node->get_logger(), "[Result] Extrinsic parameters T_cam_lidar:");
  std::cout << BOLDCYAN << std::fixed << std::setprecision(6) << transformation << RESET << std::endl;

  const Eigen::Matrix4f &projection_transformation =
      (params.use_external_extrinsic_for_coloring && external_loaded) ? external_transformation : transformation;
  if (params.use_external_extrinsic_for_coloring && external_loaded)
  {
    RCLCPP_INFO(node->get_logger(), "[Result] Coloring uses external extrinsic from: %s",
                params.external_calib_result_path.c_str());
  }
  else if (params.use_external_extrinsic_for_coloring && !external_loaded)
  {
    RCLCPP_WARN(node->get_logger(), "[Warn] Coloring fallback to current-bag extrinsic because external extrinsic is unavailable.");
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  projectPointCloudToImage(cloud_xyz, projection_transformation, qrDetectPtr->cameraMatrix_, qrDetectPtr->distCoeffs_, img_input, colored_cloud);
  cv::Mat overlay_image;
  projectPointCloudOverlayImage(cloud_xyz, projection_transformation, qrDetectPtr->cameraMatrix_, qrDetectPtr->distCoeffs_, img_input, overlay_image, 2);

  saveCalibrationResults(params, transformation, colored_cloud, qrDetectPtr->imageCopy_,
                         rmse, qrDetectPtr->reprojection_rmse_px_, external_rmse_m);

  // 输出相机与 LiDAR 拟合的圆心三维坐标到 output/circle_centers.txt
  saveCircleCenters(params.output_path, qr_centers, lidar_centers);

  auto colored_cloud_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("colored_cloud", 1);
  auto aligned_lidar_centers_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_lidar_centers", 1);
  auto overlay_image_pub = node->create_publisher<sensor_msgs::msg::Image>("overlay_image", 1);

  // 主循环
  rclcpp::Rate rate(1);
  while (rclcpp::ok())
  {
    if (DEBUG)
    {
      // 发布 QR 检测结果
      sensor_msgs::msg::PointCloud2 qr_centers_msg;
      pcl::toROSMsg(*qr_centers, qr_centers_msg);
      qr_centers_msg.header.stamp = node->now();
      qr_centers_msg.header.frame_id = "map";
      qrDetectPtr->qr_pub_->publish(qr_centers_msg);

      // 发布 LiDAR 检测结果
      sensor_msgs::msg::PointCloud2 lidar_centers_msg;
      pcl::toROSMsg(*lidar_centers, lidar_centers_msg);
      lidar_centers_msg.header = qr_centers_msg.header;
      lidarDetectPtr->center_pub_->publish(lidar_centers_msg);

      // 发布中间结果
      sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
      pcl::toROSMsg(*lidarDetectPtr->getFilteredCloud(), filtered_cloud_msg);
      filtered_cloud_msg.header = qr_centers_msg.header;
      lidarDetectPtr->filtered_pub_->publish(filtered_cloud_msg);

      sensor_msgs::msg::PointCloud2 plane_cloud_msg;
      pcl::toROSMsg(*lidarDetectPtr->getPlaneCloud(), plane_cloud_msg);
      plane_cloud_msg.header = qr_centers_msg.header;
      lidarDetectPtr->plane_pub_->publish(plane_cloud_msg);

      sensor_msgs::msg::PointCloud2 aligned_cloud_msg;
      pcl::toROSMsg(*lidarDetectPtr->getAlignedCloud(), aligned_cloud_msg);
      aligned_cloud_msg.header = qr_centers_msg.header;
      lidarDetectPtr->aligned_pub_->publish(aligned_cloud_msg);

      sensor_msgs::msg::PointCloud2 edge_cloud_msg;
      pcl::toROSMsg(*lidarDetectPtr->getEdgeCloud(), edge_cloud_msg);
      edge_cloud_msg.header = qr_centers_msg.header;
      lidarDetectPtr->edge_pub_->publish(edge_cloud_msg);

      sensor_msgs::msg::PointCloud2 lidar_centers_z0_msg;
      pcl::toROSMsg(*lidarDetectPtr->getCenterZ0Cloud(), lidar_centers_z0_msg);
      lidar_centers_z0_msg.header = qr_centers_msg.header;
      lidarDetectPtr->center_z0_pub_->publish(lidar_centers_z0_msg);

      // 发布外参变换后的LiDAR点云
      sensor_msgs::msg::PointCloud2 aligned_lidar_centers_msg;
      pcl::toROSMsg(*aligned_lidar_centers, aligned_lidar_centers_msg);
      aligned_lidar_centers_msg.header = qr_centers_msg.header;
      aligned_lidar_centers_pub->publish(aligned_lidar_centers_msg);

      // 发布彩色点云
      sensor_msgs::msg::PointCloud2 colored_cloud_msg;
      pcl::toROSMsg(*colored_cloud, colored_cloud_msg);
      colored_cloud_msg.header = qr_centers_msg.header;
      colored_cloud_pub->publish(colored_cloud_msg);

      if (!overlay_image.empty())
      {
        sensor_msgs::msg::Image overlay_msg;
        overlay_msg.header.stamp = qr_centers_msg.header.stamp;
        overlay_msg.header.frame_id = "camera";
        overlay_msg.height = overlay_image.rows;
        overlay_msg.width = overlay_image.cols;
        overlay_msg.encoding = "bgr8";
        overlay_msg.is_bigendian = false;
        overlay_msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(overlay_image.step);
        const size_t data_size = overlay_msg.step * overlay_msg.height;
        overlay_msg.data.resize(data_size);
        std::memcpy(overlay_msg.data.data(), overlay_image.data, data_size);
        overlay_image_pub->publish(overlay_msg);
      }

      // cv::imshow("result", qrDetectPtr->imageCopy_);
    }
    // cv::waitKey(1);
    rclcpp::spin_some(node);
    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
