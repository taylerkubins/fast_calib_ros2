/*
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "color.h"

using namespace std;
using namespace cv;
using namespace pcl;

#define TARGET_NUM_CIRCLES 4
#define DEBUG 1
#define GEOMETRY_TOLERANCE 0.06

// namespace CommonLiDAR
// {
//   struct EIGEN_ALIGN16 Point
//   {
//     PCL_ADD_POINT4D;     // quad-word XYZ
//     float intensity;     ///< laser intensity reading
//     std::uint16_t ring;  ///< laser ring number
//     float range;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment
//   };

//   void addRange(pcl::PointCloud<CommonLiDAR::Point> &pc)
//   {
//     for (pcl::PointCloud<Point>::iterator pt = pc.points.begin();
//          pt < pc.points.end(); pt++) {
//       pt->range = sqrt(pt->x * pt->x + pt->y * pt->y + pt->z * pt->z);
//     }
//   }

//   vector<vector<Point *>> getRings(pcl::PointCloud<CommonLiDAR::Point> &pc,
//                                    int rings_count)
//   {
//     vector<vector<Point *>> rings(rings_count);
//     for (pcl::PointCloud<Point>::iterator pt = pc.points.begin();
//          pt < pc.points.end(); pt++) {
//       rings[pt->ring].push_back(&(*pt));
//     }
//     return rings;
//   }
// }  // namespace Ouster

// POINT_CLOUD_REGISTER_POINT_STRUCT(CommonLiDAR::Point,
//                                   (float, x, x)(float, y, y)(float, z, z)(
//                                       float, intensity,
//                                       intensity)(std::uint16_t, ring,
//                                                   ring)(float, range, range));

// 参数结构体
struct Params
{
  double x_min, x_max, y_min, y_max, z_min, z_max;
  double fx, fy, cx, cy, k1, k2, p1, p2;
  double marker_size, delta_width_qr_center, delta_height_qr_center;
  double delta_width_circles, delta_height_circles, circle_radius;
  double circle_fit_error_threshold; // LiDAR 圆拟合可接受误差阈值，小于该值的圆心才被接受（默认 0.02）
  double circle_center_merge_distance; // 两圆心在平面内距离小于此值视为同一圆，保留误差更小者（默认 0.03 m）
  int edge_cluster_min_size;          // 边缘聚类最小点数，过大会漏掉点少的圆孔（默认 30）
  int min_detected_markers;
  string image_path;
  string bag_path;
  string lidar_topic;
  string output_path;
  bool use_external_extrinsic_eval;   // 是否使用外源外参做额外误差评估（默认 false，不影响原流程）
  bool use_external_extrinsic_for_coloring; // 是否使用外源外参进行点云着色与叠加投影（默认 false）
  string external_calib_result_path;  // 外源 calib_result.txt 路径
  // LiDAR 与相机轴安装映射：相机各轴由哪条 LiDAR 轴及符号得到
  // 例如 "x" 表示 +lidar_x, "-y" 表示 -lidar_y。用于 sortPatternCenters 中的坐标变换
  string lidar_axis_cam_x; // 相机 X = ?*LiDAR_?，默认 "-y"
  string lidar_axis_cam_y; // 相机 Y = ?*LiDAR_?，默认 "-z"
  string lidar_axis_cam_z; // 相机 Z = ?*LiDAR_?，默认 "x"
};

// 读取参数
Params loadParameters(rclcpp::Node::SharedPtr node)
{
  Params params;
  node->declare_parameter("fx", 1215.31801774424);
  node->declare_parameter("fy", 1214.72961288138);
  node->declare_parameter("cx", 1047.86571859677);
  node->declare_parameter("cy", 745.068353101898);
  node->declare_parameter("k1", -0.33574781188503);
  node->declare_parameter("k2", 0.10996870793601);
  node->declare_parameter("p1", 0.000157303079833973);
  node->declare_parameter("p2", 0.000544930726278493);
  node->declare_parameter("marker_size", 0.2);
  node->declare_parameter("delta_width_qr_center", 0.55);
  node->declare_parameter("delta_height_qr_center", 0.35);
  node->declare_parameter("delta_width_circles", 0.5);
  node->declare_parameter("delta_height_circles", 0.4);
  node->declare_parameter("min_detected_markers", 3);
  node->declare_parameter("circle_radius", 0.12);
  node->declare_parameter("circle_fit_error_threshold", 0.02);
  node->declare_parameter("circle_center_merge_distance", 0.03);
  node->declare_parameter("edge_cluster_min_size", 30);
  node->declare_parameter("image_path", string("/home/chunran/calib_ws/src/fast_calib/data/image.png"));
  node->declare_parameter("bag_path", string("/home/chunran/calib_ws/src/fast_calib/data/input.bag"));
  node->declare_parameter("lidar_topic", string("/livox/lidar"));
  node->declare_parameter("output_path", string("/home/chunran/calib_ws/src/fast_calib/output"));
  node->declare_parameter("x_min", 1.5);
  node->declare_parameter("x_max", 3.0);
  node->declare_parameter("y_min", -1.5);
  node->declare_parameter("y_max", 2.0);
  node->declare_parameter("z_min", -0.5);
  node->declare_parameter("z_max", 2.0);
  node->declare_parameter("lidar_axis_cam_x", string("-y"));
  node->declare_parameter("lidar_axis_cam_y", string("-z"));
  node->declare_parameter("lidar_axis_cam_z", string("x"));
  node->declare_parameter("use_external_extrinsic_eval", false);
  node->declare_parameter("use_external_extrinsic_for_coloring", false);
  node->declare_parameter("external_calib_result_path", string(""));

  node->get_parameter_or("fx", params.fx, 1215.31801774424);
  node->get_parameter_or("fy", params.fy, 1214.72961288138);
  node->get_parameter_or("cx", params.cx, 1047.86571859677);
  node->get_parameter_or("cy", params.cy, 745.068353101898);
  node->get_parameter_or("k1", params.k1, -0.33574781188503);
  node->get_parameter_or("k2", params.k2, 0.10996870793601);
  node->get_parameter_or("p1", params.p1, 0.000157303079833973);
  node->get_parameter_or("p2", params.p2, 0.000544930726278493);
  node->get_parameter_or("marker_size", params.marker_size, 0.2);
  node->get_parameter_or("delta_width_qr_center", params.delta_width_qr_center, 0.55);
  node->get_parameter_or("delta_height_qr_center", params.delta_height_qr_center, 0.35);
  node->get_parameter_or("delta_width_circles", params.delta_width_circles, 0.5);
  node->get_parameter_or("delta_height_circles", params.delta_height_circles, 0.4);
  node->get_parameter_or("min_detected_markers", params.min_detected_markers, 3);
  node->get_parameter_or("circle_radius", params.circle_radius, 0.12);
  node->get_parameter_or("circle_fit_error_threshold", params.circle_fit_error_threshold, 0.02);
  node->get_parameter_or("circle_center_merge_distance", params.circle_center_merge_distance, 0.03);
  node->get_parameter_or("edge_cluster_min_size", params.edge_cluster_min_size, 30);
  node->get_parameter_or("image_path", params.image_path, string("/home/chunran/calib_ws/src/fast_calib/data/image.png"));
  node->get_parameter_or("bag_path", params.bag_path, string("/home/chunran/calib_ws/src/fast_calib/data/input.bag"));
  node->get_parameter_or("lidar_topic", params.lidar_topic, string("/livox/lidar"));
  node->get_parameter_or("output_path", params.output_path, string("/home/chunran/calib_ws/src/fast_calib/output"));
  node->get_parameter_or("x_min", params.x_min, 1.5);
  node->get_parameter_or("x_max", params.x_max, 3.0);
  node->get_parameter_or("y_min", params.y_min, -1.5);
  node->get_parameter_or("y_max", params.y_max, 2.0);
  node->get_parameter_or("z_min", params.z_min, -0.5);
  node->get_parameter_or("z_max", params.z_max, 2.0);
  node->get_parameter_or("lidar_axis_cam_x", params.lidar_axis_cam_x, string("-y"));
  node->get_parameter_or("lidar_axis_cam_y", params.lidar_axis_cam_y, string("-z"));
  node->get_parameter_or("lidar_axis_cam_z", params.lidar_axis_cam_z, string("x"));
  node->get_parameter_or("use_external_extrinsic_eval", params.use_external_extrinsic_eval, false);
  node->get_parameter_or("use_external_extrinsic_for_coloring", params.use_external_extrinsic_for_coloring, false);
  node->get_parameter_or("external_calib_result_path", params.external_calib_result_path, string(""));
  return params;
}

inline std::vector<float> parseNumbersFromText(const std::string &text)
{
  std::string cleaned = text;
  for (char &c : cleaned)
  {
    const bool keep = std::isdigit(static_cast<unsigned char>(c)) || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E';
    if (!keep) c = ' ';
  }
  std::stringstream ss(cleaned);
  std::vector<float> values;
  float v = 0.0f;
  while (ss >> v)
    values.push_back(v);
  return values;
}

inline bool loadExtrinsicFromCalibResult(const std::string &calib_result_path, Eigen::Matrix4f &transformation)
{
  std::ifstream in(calib_result_path);
  if (!in.is_open())
    return false;

  std::vector<float> r_vals;
  std::vector<float> p_vals;
  std::string line;
  bool reading_r = false;
  while (std::getline(in, line))
  {
    if (line.find("Rcl:") != std::string::npos)
    {
      reading_r = true;
    }
    if (reading_r && r_vals.size() < 9)
    {
      std::vector<float> nums = parseNumbersFromText(line);
      r_vals.insert(r_vals.end(), nums.begin(), nums.end());
      if (r_vals.size() >= 9)
        reading_r = false;
      continue;
    }

    if (line.find("Pcl:") != std::string::npos)
    {
      p_vals = parseNumbersFromText(line);
      break;
    }
  }

  if (r_vals.size() < 9 || p_vals.size() < 3)
    return false;

  transformation = Eigen::Matrix4f::Identity();
  transformation(0, 0) = r_vals[0];
  transformation(0, 1) = r_vals[1];
  transformation(0, 2) = r_vals[2];
  transformation(1, 0) = r_vals[3];
  transformation(1, 1) = r_vals[4];
  transformation(1, 2) = r_vals[5];
  transformation(2, 0) = r_vals[6];
  transformation(2, 1) = r_vals[7];
  transformation(2, 2) = r_vals[8];
  transformation(0, 3) = p_vals[0];
  transformation(1, 3) = p_vals[1];
  transformation(2, 3) = p_vals[2];
  return true;
}

double computeRMSE(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud1,
                   const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud2)
{
  if (cloud1->size() != cloud2->size())
  {
    std::cerr << BOLDRED << "[computeRMSE] Point cloud sizes do not match, cannot compute RMSE." << RESET << std::endl;
    return -1.0;
  }

  double sum = 0.0;
  for (size_t i = 0; i < cloud1->size(); ++i)
  {
    double dx = cloud1->points[i].x - cloud2->points[i].x;
    double dy = cloud1->points[i].y - cloud2->points[i].y;
    double dz = cloud1->points[i].z - cloud2->points[i].z;

    sum += dx * dx + dy * dy + dz * dz;
  }

  double mse = sum / cloud1->size();
  return std::sqrt(mse);
}

// 将 LiDAR 点云转换到 QR 码坐标系
void alignPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud, const Eigen::Matrix4f &transformation)
{
  output_cloud->clear();
  for (const auto &pt : input_cloud->points)
  {
    Eigen::Vector4f pt_homogeneous(pt.x, pt.y, pt.z, 1.0);
    Eigen::Vector4f transformed_pt = transformation * pt_homogeneous;
    output_cloud->push_back(pcl::PointXYZ(transformed_pt(0), transformed_pt(1), transformed_pt(2)));
  }
}

void projectPointCloudToImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                              const Eigen::Matrix4f &transformation,
                              const cv::Mat &cameraMatrix,
                              const cv::Mat &distCoeffs,
                              const cv::Mat &image,
                              pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud)
{
  colored_cloud->clear();
  colored_cloud->reserve(cloud->size());

  // Undistort the entire image (preprocess outside if possible)
  cv::Mat undistortedImage;
  cv::undistort(image, undistortedImage, cameraMatrix, distCoeffs);

  // Precompute rotation and translation vectors (zero for this case)
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_32F);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F);
  cv::Mat zeroDistCoeffs = cv::Mat::zeros(5, 1, CV_32F);

  // Preallocate memory for projection
  std::vector<cv::Point3f> objectPoints(1);
  std::vector<cv::Point2f> imagePoints(1);

  for (const auto &point : *cloud)
  {
    // Transform the point
    Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
    Eigen::Vector4f transformed_point = transformation * homogeneous_point;

    // Skip points behind the camera
    if (transformed_point(2) < 0)
      continue;

    // Project the point to the image plane
    objectPoints[0] = cv::Point3f(transformed_point(0), transformed_point(1), transformed_point(2));
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, zeroDistCoeffs, imagePoints);

    int u = static_cast<int>(imagePoints[0].x);
    int v = static_cast<int>(imagePoints[0].y);

    // Check if the point is within the image bounds
    if (u >= 0 && u < undistortedImage.cols && v >= 0 && v < undistortedImage.rows)
    {
      // Get the color from the undistorted image
      cv::Vec3b color = undistortedImage.at<cv::Vec3b>(v, u);

      // Create a colored point and add it to the cloud
      pcl::PointXYZRGB colored_point;
      colored_point.x = transformed_point(0);
      colored_point.y = transformed_point(1);
      colored_point.z = transformed_point(2);
      colored_point.r = color[2];
      colored_point.g = color[1];
      colored_point.b = color[0];
      colored_cloud->push_back(colored_point);
    }
  }
}

void projectPointCloudOverlayImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                   const Eigen::Matrix4f &transformation,
                                   const cv::Mat &cameraMatrix,
                                   const cv::Mat &distCoeffs,
                                   const cv::Mat &image,
                                   cv::Mat &overlay_image,
                                   int point_radius = 1)
{
  cv::Mat undistortedImage;
  cv::undistort(image, undistortedImage, cameraMatrix, distCoeffs);
  overlay_image = undistortedImage.clone();

  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_32F);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F);
  cv::Mat zeroDistCoeffs = cv::Mat::zeros(5, 1, CV_32F);

  std::vector<cv::Point3f> objectPoints(1);
  std::vector<cv::Point2f> imagePoints(1);

  for (const auto &point : *cloud)
  {
    Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
    Eigen::Vector4f transformed_point = transformation * homogeneous_point;

    if (transformed_point(2) < 0)
      continue;

    objectPoints[0] = cv::Point3f(transformed_point(0), transformed_point(1), transformed_point(2));
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, zeroDistCoeffs, imagePoints);

    int u = static_cast<int>(imagePoints[0].x);
    int v = static_cast<int>(imagePoints[0].y);
    if (u < 0 || u >= overlay_image.cols || v < 0 || v >= overlay_image.rows)
      continue;

    // Color by depth in camera frame (near=red, far=blue).
    float depth = transformed_point(2);
    float t = std::max(0.0f, std::min(1.0f, depth / 20.0f));
    cv::Scalar color(255.0 * t, 255.0 * (1.0f - t), 255.0 * (1.0f - t));
    cv::circle(overlay_image, cv::Point(u, v), point_radius, color, -1, cv::LINE_AA);
  }
}

void saveCalibrationResults(const Params &params, const Eigen::Matrix4f &transformation,
                            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud, const cv::Mat &img_input,
                            double rmse_m = -1.0, double reprojection_rmse_px = -1.0,
                            double external_rmse_m = -1.0)
{
  if (colored_cloud->empty())
  {
    std::cerr << BOLDRED << "[saveCalibrationResults] Colored point cloud is empty!" << RESET << std::endl;
    return;
  }
  std::string outputDir = params.output_path;
  if (outputDir.back() != '/')
    outputDir += '/';

  std::ofstream outFile(outputDir + "calib_result.txt");
  if (outFile.is_open())
  {
    outFile << "# FAST-LIVO2 calibration format\n";
    outFile << "cam_model: Pinhole\n";
    outFile << "cam_width: " << img_input.cols << "\n";
    outFile << "cam_height: " << img_input.rows << "\n";
    outFile << "scale: 1.0\n";
    outFile << "cam_fx: " << params.fx << "\n";
    outFile << "cam_fy: " << params.fy << "\n";
    outFile << "cam_cx: " << params.cx << "\n";
    outFile << "cam_cy: " << params.cy << "\n";
    outFile << "cam_d0: " << params.k1 << "\n";
    outFile << "cam_d1: " << params.k2 << "\n";
    outFile << "cam_d2: " << params.p1 << "\n";
    outFile << "cam_d3: " << params.p2 << "\n";
    if (rmse_m >= 0.0)
      outFile << "rmse_m: " << std::fixed << std::setprecision(6) << rmse_m << "\n";
    if (reprojection_rmse_px >= 0.0)
      outFile << "reprojection_rmse_px: " << std::fixed << std::setprecision(6) << reprojection_rmse_px << "\n";
    if (external_rmse_m >= 0.0)
      outFile << "external_rmse_m: " << std::fixed << std::setprecision(6) << external_rmse_m << "\n";

    outFile << "\nRcl: [" << std::fixed << std::setprecision(6);
    outFile << std::setw(10) << transformation(0, 0) << ", " << std::setw(10) << transformation(0, 1) << ", " << std::setw(10) << transformation(0, 2) << ",\n";
    outFile << "      " << std::setw(10) << transformation(1, 0) << ", " << std::setw(10) << transformation(1, 1) << ", " << std::setw(10) << transformation(1, 2) << ",\n";
    outFile << "      " << std::setw(10) << transformation(2, 0) << ", " << std::setw(10) << transformation(2, 1) << ", " << std::setw(10) << transformation(2, 2) << "]\n";

    outFile << "Pcl: [";
    outFile << std::setw(10) << transformation(0, 3) << ", " << std::setw(10) << transformation(1, 3) << ", " << std::setw(10) << transformation(2, 3) << "]\n";

    outFile.close();
    std::cout << BOLDYELLOW << "[Result] Calibration results saved to " << BOLDWHITE << outputDir << "calib_result.txt" << RESET << std::endl;
  }
  else
  {
    std::cerr << BOLDRED << "[Error] Failed to open calib_result.txt for writing!" << RESET << std::endl;
  }

  if (pcl::io::savePCDFileASCII(outputDir + "colored_cloud.pcd", *colored_cloud) == 0)
  {
    pcl::io::savePLYFile(outputDir + "colored_cloud.ply", *colored_cloud);
    std::cout << BOLDYELLOW << "[Result] Saved colored point cloud to: " << BOLDWHITE << outputDir << "colored_cloud.pcd" << RESET << std::endl;
  }
  else
  {
    std::cerr << BOLDRED << "[Error] Failed to save colored point cloud to " << outputDir << "colored_cloud.pcd" << "!" << RESET << std::endl;
  }

  imwrite(outputDir + "qr_detect.png", img_input);
}

// 将相机与 LiDAR 拟合的圆心三维坐标保存到 output 目录下的 .txt 文件
inline void saveCircleCenters(const std::string &outputDir,
                              const pcl::PointCloud<pcl::PointXYZ>::Ptr &camera_centers,
                              const pcl::PointCloud<pcl::PointXYZ>::Ptr &lidar_centers)
{
  std::string dir = outputDir;
  if (!dir.empty() && dir.back() != '/')
    dir += '/';

  std::ofstream out(dir + "circle_centers.txt");
  if (!out.is_open())
  {
    std::cerr << BOLDRED << "[saveCircleCenters] Failed to open " << dir << "circle_centers.txt" << RESET << std::endl;
    return;
  }
  out << std::fixed << std::setprecision(6);
  out << "# Circle centers from calibration (same order for SVD correspondence)\n";
  out << "# Camera frame: 4 points (x y z in meters)\n";
  for (size_t i = 0; i < camera_centers->size(); ++i)
    out << camera_centers->points[i].x << " " << camera_centers->points[i].y << " " << camera_centers->points[i].z << "\n";
  out << "# LiDAR frame: 4 points (x y z in meters)\n";
  for (size_t i = 0; i < lidar_centers->size(); ++i)
    out << lidar_centers->points[i].x << " " << lidar_centers->points[i].y << " " << lidar_centers->points[i].z << "\n";
  out.close();
  std::cout << BOLDYELLOW << "[Result] Circle centers saved to " << BOLDWHITE << dir << "circle_centers.txt" << RESET << std::endl;
}

// 解析轴映射字符串 "x"/"-x"/"y"/"-y"/"z"/"-z" -> (轴索引 0=x,1=y,2=z, 符号 ±1)。解析失败返回 (0, 0) 并打日志
inline void parseAxisMapping(const std::string &s, int &axis_index, int &sign)
{
  axis_index = 0;
  sign = 1;
  if (s.empty()) return;
  size_t i = 0;
  if (s[0] == '-') { sign = -1; i = 1; }
  else if (s[0] == '+') { i = 1; }
  if (i >= s.size()) return;
  if (s[i] == 'x') axis_index = 0;
  else if (s[i] == 'y') axis_index = 1;
  else if (s[i] == 'z') axis_index = 2;
}

// 从 LiDAR 点 p 取第 axis_index 分量 (0=x,1=y,2=z)
inline float getLidarComponent(const pcl::PointXYZ &p, int axis_index)
{
  if (axis_index == 0) return p.x;
  if (axis_index == 1) return p.y;
  return p.z;
}

void sortPatternCenters(pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr v,
                        const std::string &axis_mode = "camera",
                        const Params *params = nullptr)
{
  if (pc->size() != 4)
  {
    std::cerr << "Number of centers must be 4\n";
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr work_pc(new pcl::PointCloud<pcl::PointXYZ>());

  // 可配置的 LiDAR -> 相机 轴变换（用于排序时统一到“类相机”坐标系）
  std::string ax_x = "-y", ax_y = "-z", ax_z = "x";
  if (params)
  {
    ax_x = params->lidar_axis_cam_x;
    ax_y = params->lidar_axis_cam_y;
    ax_z = params->lidar_axis_cam_z;
  }

  if (axis_mode == "lidar")
  {
    int ax_ix, ax_iy, ax_iz, sx, sy, sz;
    parseAxisMapping(ax_x, ax_ix, sx);
    parseAxisMapping(ax_y, ax_iy, sy);
    parseAxisMapping(ax_z, ax_iz, sz);
    for (const auto &p : *pc)
    {
      pcl::PointXYZ pt;
      pt.x = sx * getLidarComponent(p, ax_ix);
      pt.y = sy * getLidarComponent(p, ax_iy);
      pt.z = sz * getLidarComponent(p, ax_iz);
      work_pc->push_back(pt);
    }
  }
  else
  {
    *work_pc = *pc;
  }

  // --- Sorting based on the local coordinate system of the pattern ---
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*work_pc, centroid);
  pcl::PointXYZ ref_origin(centroid[0], centroid[1], centroid[2]);

  std::vector<std::pair<float, int>> proj_points;
  for (size_t i = 0; i < work_pc->size(); ++i)
  {
    const auto &p = work_pc->points[i];
    Eigen::Vector3f rel_vec(p.x - ref_origin.x, p.y - ref_origin.y, p.z - ref_origin.z);
    proj_points.emplace_back(atan2(rel_vec.y(), rel_vec.x()), i);
  }
  std::sort(proj_points.begin(), proj_points.end());

  v->resize(4);
  for (int i = 0; i < 4; ++i)
    (*v)[i] = work_pc->points[proj_points[i].second];

  const auto &p0 = v->points[0];
  const auto &p1 = v->points[1];
  const auto &p2 = v->points[2];
  Eigen::Vector3f v01(p1.x - p0.x, p1.y - p0.y, 0);
  Eigen::Vector3f v12(p2.x - p1.x, p2.y - p1.y, 0);
  if (v01.cross(v12).z() > 0)
    std::swap((*v)[1], (*v)[3]);

  // 将排序后的点从“类相机”系变回 LiDAR 系：逆映射
  if (axis_mode == "lidar")
  {
    int ax_ix, ax_iy, ax_iz, sx, sy, sz;
    parseAxisMapping(ax_x, ax_ix, sx);
    parseAxisMapping(ax_y, ax_iy, sy);
    parseAxisMapping(ax_z, ax_iz, sz);
    int inv_axis[3], inv_sign[3];
    if (ax_ix == 0) { inv_axis[0] = 0; inv_sign[0] = sx; }
    else if (ax_iy == 0) { inv_axis[0] = 1; inv_sign[0] = sy; }
    else { inv_axis[0] = 2; inv_sign[0] = sz; }
    if (ax_ix == 1) { inv_axis[1] = 0; inv_sign[1] = sx; }
    else if (ax_iy == 1) { inv_axis[1] = 1; inv_sign[1] = sy; }
    else { inv_axis[1] = 2; inv_sign[1] = sz; }
    if (ax_ix == 2) { inv_axis[2] = 0; inv_sign[2] = sx; }
    else if (ax_iy == 2) { inv_axis[2] = 1; inv_sign[2] = sy; }
    else { inv_axis[2] = 2; inv_sign[2] = sz; }
    for (auto &point : v->points)
    {
      float cam[3] = { point.x, point.y, point.z };
      point.x = inv_sign[0] * cam[inv_axis[0]];
      point.y = inv_sign[1] * cam[inv_axis[1]];
      point.z = inv_sign[2] * cam[inv_axis[2]];
    }
  }
}

class Square
{
private:
  pcl::PointXYZ _center;
  std::vector<pcl::PointXYZ> _candidates;
  float _target_width, _target_height, _target_diagonal;

public:
  Square(std::vector<pcl::PointXYZ> candidates, float width, float height)
  {
    _candidates = candidates;
    _target_width = width;
    _target_height = height;
    _target_diagonal = sqrt(pow(width, 2) + pow(height, 2));

    // Compute candidates centroid
    _center.x = _center.y = _center.z = 0;
    for (int i = 0; i < candidates.size(); ++i)
    {
      _center.x += candidates[i].x;
      _center.y += candidates[i].y;
      _center.z += candidates[i].z;
    }

    _center.x /= candidates.size();
    _center.y /= candidates.size();
    _center.z /= candidates.size();
  }

  float distance(pcl::PointXYZ pt1, pcl::PointXYZ pt2)
  {
    return sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2) +
                pow(pt1.z - pt2.z, 2));
  }

  pcl::PointXYZ at(int i)
  {
    assert(0 <= i && i < 4);
    return _candidates[i];
  }

  // ==================================================================================================
  // The original is_valid() was too rigid. This version is more robust by checking for two possible
  // orderings of the side lengths (width-height vs. height-width) after angular sorting.
  // ==================================================================================================
  bool is_valid()
  {
    if (_candidates.size() != 4)
      return false;

    pcl::PointCloud<pcl::PointXYZ>::Ptr candidates_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto &p : _candidates)
      candidates_cloud->push_back(p);

    // Check if candidates are at a reasonable distance from their centroid
    for (int i = 0; i < _candidates.size(); ++i)
    {
      float d = distance(_center, _candidates[i]);
      // Check if distance from center to corner is close to half the diagonal length
      if (fabs(d - _target_diagonal / 2.) / (_target_diagonal / 2.) > GEOMETRY_TOLERANCE * 2.0)
      { // Loosened tolerance slightly
        // std::cout << "[Debug] Corner distance from center failed." << std::endl;
        return false;
      }
    }

    // Sort the corners counter-clockwise
    pcl::PointCloud<pcl::PointXYZ>::Ptr sorted_centers(new pcl::PointCloud<pcl::PointXYZ>());
    sortPatternCenters(candidates_cloud, sorted_centers, "camera");

    // Get the four side lengths from the sorted points
    float s01 = distance(sorted_centers->points[0], sorted_centers->points[1]);
    float s12 = distance(sorted_centers->points[1], sorted_centers->points[2]);
    float s23 = distance(sorted_centers->points[2], sorted_centers->points[3]);
    float s30 = distance(sorted_centers->points[3], sorted_centers->points[0]);

    // Check for pattern 1: width, height, width, height
    bool pattern1_ok =
        (fabs(s01 - _target_width) / _target_width < GEOMETRY_TOLERANCE) &&
        (fabs(s12 - _target_height) / _target_height < GEOMETRY_TOLERANCE) &&
        (fabs(s23 - _target_width) / _target_width < GEOMETRY_TOLERANCE) &&
        (fabs(s30 - _target_height) / _target_height < GEOMETRY_TOLERANCE);

    // Check for pattern 2: height, width, height, width
    bool pattern2_ok =
        (fabs(s01 - _target_height) / _target_height < GEOMETRY_TOLERANCE) &&
        (fabs(s12 - _target_width) / _target_width < GEOMETRY_TOLERANCE) &&
        (fabs(s23 - _target_height) / _target_height < GEOMETRY_TOLERANCE) &&
        (fabs(s30 - _target_width) / _target_width < GEOMETRY_TOLERANCE);

    if (!pattern1_ok && !pattern2_ok)
    {
      // std::cout << "[Debug] Side length geometry validation failed. Measured sides: "
      //           << s01 << ", " << s12 << ", " << s23 << ", " << s30 << std::endl;
      // std::cout << "[Debug] Target width/height: " << _target_width << " / " << _target_height << std::endl;
      return false;
    }

    // Final check on perimeter
    float perimeter = s01 + s12 + s23 + s30;
    float ideal_perimeter = 2 * (_target_width + _target_height);
    if (fabs(perimeter - ideal_perimeter) / ideal_perimeter > GEOMETRY_TOLERANCE)
    {
      // std::cout << "[Debug] Perimeter check failed. Measured: " << perimeter << ", Ideal: " << ideal_perimeter << std::endl;
      return false;
    }

    return true;
  }
};

void save2PLY(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string &filename)
{
  if (DEBUG)
  {
    pcl::io::savePLYFile(filename, *cloud);
  }
}
#endif
