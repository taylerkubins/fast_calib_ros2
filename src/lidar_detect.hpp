/*
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIDAR_DETECT_HPP
#define LIDAR_DETECT_HPP
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/io.h>
#include <deque>
#include <unordered_map>
#include "common_lib.h"

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_, delta_width_circles_, delta_height_circles_;
    rclcpp::Node::SharedPtr node_;
    std::string output_path_;

    pcl::PointCloud<Common::Point>::Ptr filtered_cloud_;
    pcl::PointCloud<Common::Point>::Ptr plane_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_z0_cloud_;

    void cropROI(pcl::PointCloud<Common::Point>::Ptr cloud)
    {
        filtered_cloud_->clear();
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);
        pass_x.filter(*filtered_cloud_);

        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);
        pass_y.filter(*filtered_cloud_);

        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);
        pass_z.filter(*filtered_cloud_);
    }

    Eigen::Matrix3d rotationAligningNormalToZ(const pcl::ModelCoefficients::Ptr &plane_coefficients,
                                              Eigen::Vector3d &normal) const
    {
        normal = Eigen::Vector3d(plane_coefficients->values[0],
                                 plane_coefficients->values[1],
                                 plane_coefficients->values[2]);
        normal.normalize();
        Eigen::Vector3d z_axis(0, 0, 1);
        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(std::max(-1.0, std::min(1.0, normal.dot(z_axis))));
        if (axis.norm() < 1e-8)
        {
            axis = Eigen::Vector3d(1, 0, 0);
            angle = 0.0;
        }
        else
        {
            axis.normalize();
        }
        return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    }

    // ROS1 detect_mech_lidar 的核心：在对齐后的边缘点上反复 RANSAC 拟合 CIRCLE2D
    void fitCirclesIterative(pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud)
    {
        center_z0_cloud_->clear();
        RCLCPP_INFO(node_->get_logger(), "[LiDAR] Start iterative circle detection, cloud size = %zu",
                    xy_cloud->points.size());

        pcl::SACSegmentation<pcl::PointXYZ> circle_segmentation;
        circle_segmentation.setModelType(pcl::SACMODEL_CIRCLE2D);
        circle_segmentation.setMethodType(pcl::SAC_RANSAC);
        circle_segmentation.setDistanceThreshold(0.02);
        circle_segmentation.setOptimizeCoefficients(true);
        circle_segmentation.setMaxIterations(1000);
        circle_segmentation.setRadiusLimits(circle_radius_ - 0.03, circle_radius_ + 0.03);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ExtractIndices<pcl::PointXYZ> extract2;

        while (xy_cloud->points.size() > 3)
        {
            circle_segmentation.setInputCloud(xy_cloud);
            inliers->indices.clear();
            circle_segmentation.segment(*inliers, *coefficients);

            if (inliers->indices.empty())
            {
                RCLCPP_INFO(node_->get_logger(), "[LiDAR] No more circles can be found, stop.");
                break;
            }
            if (static_cast<int>(inliers->indices.size()) < 5)
            {
                RCLCPP_INFO(node_->get_logger(), "[LiDAR] Found circle but inliers too few (%zu), stop.",
                            inliers->indices.size());
                break;
            }

            pcl::PointXYZ center_point;
            center_point.x = coefficients->values[0];
            center_point.y = coefficients->values[1];
            center_point.z = 0;
            center_z0_cloud_->push_back(center_point);
            RCLCPP_INFO(node_->get_logger(),
                        "[LiDAR] Circle %zu: center=(%.4f, %.4f) r=%.4f inliers=%zu remain=%zu",
                        center_z0_cloud_->size(), center_point.x, center_point.y,
                        coefficients->values[2], inliers->indices.size(), xy_cloud->size());

            extract2.setInputCloud(xy_cloud);
            extract2.setIndices(inliers);
            extract2.setNegative(true);
            pcl::PointCloud<pcl::PointXYZ>::Ptr remaining(new pcl::PointCloud<pcl::PointXYZ>);
            extract2.filter(*remaining);
            // 同时挖掉该圆附近的环带点，避免同一圆被反复拟合、其它圆被淹没
            pcl::PointCloud<pcl::PointXYZ>::Ptr pruned(new pcl::PointCloud<pcl::PointXYZ>);
            const double r_found = coefficients->values[2];
            const double band = std::max(0.05, r_found * 0.4);
            for (const auto &pt : remaining->points)
            {
                const double dx = pt.x - center_point.x;
                const double dy = pt.y - center_point.y;
                const double d = std::sqrt(dx * dx + dy * dy);
                if (std::fabs(d - r_found) > band && d > 0.03)
                    pruned->push_back(pt);
            }
            xy_cloud.swap(pruned);
            RCLCPP_INFO(node_->get_logger(), "[LiDAR] Remaining edge points after removal: %zu", xy_cloud->size());
        }
    }

    // 在对齐后的平面占用栅格上找内部圆孔（适合 Livox 等密集固态雷达）
    void detectHolesOnAlignedPlane()
    {
        center_z0_cloud_->clear();
        if (aligned_cloud_->empty())
            return;

        const double res = 0.01;
        float min_x = aligned_cloud_->points[0].x, max_x = min_x;
        float min_y = aligned_cloud_->points[0].y, max_y = min_y;
        for (const auto &p : aligned_cloud_->points)
        {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
        const int nx = static_cast<int>(std::floor((max_x - min_x) / res)) + 3;
        const int ny = static_cast<int>(std::floor((max_y - min_y) / res)) + 3;
        if (nx <= 4 || ny <= 4 || nx * ny > 400000)
        {
            RCLCPP_WARN(node_->get_logger(), "[LiDAR] Occupancy grid invalid: %dx%d", nx, ny);
            return;
        }

        std::vector<char> occ(static_cast<size_t>(nx) * ny, 0);
        auto at = [&](int i, int j) -> char & { return occ[static_cast<size_t>(i) * ny + j]; };
        for (const auto &p : aligned_cloud_->points)
        {
            int i = static_cast<int>(std::floor((p.x - min_x) / res)) + 1;
            int j = static_cast<int>(std::floor((p.y - min_y) / res)) + 1;
            if (i >= 0 && i < nx && j >= 0 && j < ny)
                at(i, j) = 1;
        }

        std::vector<char> background(occ.size(), 0);
        std::deque<std::pair<int, int>> q;
        auto push_bg = [&](int i, int j) {
            if (i < 0 || j < 0 || i >= nx || j >= ny)
                return;
            const size_t idx = static_cast<size_t>(i) * ny + j;
            if (occ[idx] || background[idx])
                return;
            background[idx] = 1;
            q.emplace_back(i, j);
        };
        for (int i = 0; i < nx; ++i)
        {
            push_bg(i, 0);
            push_bg(i, ny - 1);
        }
        for (int j = 0; j < ny; ++j)
        {
            push_bg(0, j);
            push_bg(nx - 1, j);
        }
        const int di[4] = {1, -1, 0, 0};
        const int dj[4] = {0, 0, 1, -1};
        while (!q.empty())
        {
            auto [i, j] = q.front();
            q.pop_front();
            for (int k = 0; k < 4; ++k)
                push_bg(i + di[k], j + dj[k]);
        }

        std::vector<char> hole(occ.size(), 0);
        for (size_t t = 0; t < occ.size(); ++t)
        {
            if (!occ[t] && !background[t])
                hole[t] = 1;
        }

        std::vector<char> visited(occ.size(), 0);
        const double expected_area = M_PI * circle_radius_ * circle_radius_;
        const double min_area = expected_area * 0.40;
        const double max_area = expected_area * 1.80;

        for (int si = 0; si < nx; ++si)
        {
            for (int sj = 0; sj < ny; ++sj)
            {
                const size_t sidx = static_cast<size_t>(si) * ny + sj;
                if (!hole[sidx] || visited[sidx])
                    continue;
                std::vector<std::pair<int, int>> cells;
                q.clear();
                q.emplace_back(si, sj);
                visited[sidx] = 1;
                while (!q.empty())
                {
                    auto [i, j] = q.front();
                    q.pop_front();
                    cells.emplace_back(i, j);
                    for (int k = 0; k < 4; ++k)
                    {
                        int ni = i + di[k], nj = j + dj[k];
                        if (ni < 0 || nj < 0 || ni >= nx || nj >= ny)
                            continue;
                        const size_t nidx = static_cast<size_t>(ni) * ny + nj;
                        if (!hole[nidx] || visited[nidx])
                            continue;
                        visited[nidx] = 1;
                        q.emplace_back(ni, nj);
                    }
                }

                const double area = static_cast<double>(cells.size()) * res * res;
                if (area < min_area || area > max_area)
                    continue;

                double sx = 0.0, sy = 0.0;
                for (const auto &c : cells)
                {
                    sx += min_x + (c.first - 0.5) * res;
                    sy += min_y + (c.second - 0.5) * res;
                }
                const double cx0 = sx / cells.size();
                const double cy0 = sy / cells.size();

                // 用圆孔轮廓点做 CIRCLE2D 精修
                pcl::PointCloud<pcl::PointXYZ>::Ptr contour(new pcl::PointCloud<pcl::PointXYZ>);
                for (const auto &c : cells)
                {
                    bool border = false;
                    for (int k = 0; k < 4; ++k)
                    {
                        int ni = c.first + di[k], nj = c.second + dj[k];
                        if (ni < 0 || nj < 0 || ni >= nx || nj >= ny || at(ni, nj))
                        {
                            border = true;
                            break;
                        }
                    }
                    if (border)
                        contour->push_back(pcl::PointXYZ(min_x + (c.first - 0.5f) * res,
                                                         min_y + (c.second - 0.5f) * res, 0.f));
                }
                // 再并入真实平面点中靠近估计圆周的点
                for (const auto &p : aligned_cloud_->points)
                {
                    const double d = std::hypot(p.x - cx0, p.y - cy0);
                    if (std::fabs(d - circle_radius_) < 0.03)
                        contour->push_back(p);
                }

                pcl::PointXYZ center;
                center.x = static_cast<float>(cx0);
                center.y = static_cast<float>(cy0);
                center.z = 0.f;
                double r_fit = std::sqrt(area / M_PI);

                if (contour->size() >= 8)
                {
                    pcl::SACSegmentation<pcl::PointXYZ> seg;
                    seg.setOptimizeCoefficients(true);
                    seg.setModelType(pcl::SACMODEL_CIRCLE2D);
                    seg.setMethodType(pcl::SAC_RANSAC);
                    seg.setDistanceThreshold(0.015);
                    seg.setMaxIterations(1000);
                    seg.setRadiusLimits(circle_radius_ - 0.04, circle_radius_ + 0.04);
                    seg.setInputCloud(contour);
                    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
                    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                    seg.segment(*inliers, *coeff);
                    if (inliers->indices.size() >= 8 && coeff->values.size() >= 3)
                    {
                        center.x = coeff->values[0];
                        center.y = coeff->values[1];
                        r_fit = coeff->values[2];
                    }
                }

                center_z0_cloud_->push_back(center);
                RCLCPP_INFO(node_->get_logger(),
                            "[LiDAR] Hole candidate %zu: center=(%.4f, %.4f) r=%.4f area=%.4f contour=%zu",
                            center_z0_cloud_->size(), center.x, center.y, r_fit, area, contour->size());
            }
        }

        // 可视化：把圆孔圆周附近的平面点当作 edge
        edge_cloud_->clear();
        for (const auto &p : aligned_cloud_->points)
        {
            for (const auto &c : center_z0_cloud_->points)
            {
                const double d = std::hypot(p.x - c.x, p.y - c.y);
                if (std::fabs(d - circle_radius_) < 0.025)
                {
                    edge_cloud_->push_back(p);
                    break;
                }
            }
        }
    }

    bool selectCentersByGeometry(const Eigen::Matrix3d &R_align, float average_z,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        std::vector<std::vector<int>> groups;
        comb(static_cast<int>(center_z0_cloud_->size()), TARGET_NUM_CIRCLES, groups);
        std::vector<double> groups_scores(groups.size(), -1.0);

        for (size_t i = 0; i < groups.size(); ++i)
        {
            std::vector<pcl::PointXYZ> candidates;
            for (int j : groups[i])
                candidates.push_back(center_z0_cloud_->at(j));
            Square square_candidate(candidates, delta_width_circles_, delta_height_circles_);
            groups_scores[i] = square_candidate.is_valid() ? 1.0 : -1.0;
        }

        int best_candidate_idx = -1;
        double best_candidate_score = -1;
        for (size_t i = 0; i < groups.size(); ++i)
        {
            if (best_candidate_score == 1 && groups_scores[i] == 1)
            {
                RCLCPP_ERROR(node_->get_logger(),
                             "[LiDAR] More than one set of candidates fit target's geometry. "
                             "Please check ROI / circle params.");
                return false;
            }
            if (groups_scores[i] > best_candidate_score)
            {
                best_candidate_score = groups_scores[i];
                best_candidate_idx = static_cast<int>(i);
            }
        }
        if (best_candidate_idx < 0)
        {
            RCLCPP_WARN(node_->get_logger(),
                        "[LiDAR] Unable to find a candidate set that matches target's geometry "
                        "(%zu candidates, w=%.2f h=%.2f)",
                        center_z0_cloud_->size(), delta_width_circles_, delta_height_circles_);
            return false;
        }

        Eigen::Matrix3d R_inv = R_align.inverse();
        for (int j : groups[best_candidate_idx])
        {
            const auto &center = center_z0_cloud_->at(j);
            Eigen::Vector3d aligned_point(center.x, center.y, center.z + average_z);
            Eigen::Vector3d original_point = R_inv * aligned_point;
            pcl::PointXYZ center_point_origin;
            center_point_origin.x = original_point.x();
            center_point_origin.y = original_point.y();
            center_point_origin.z = original_point.z();
            center_cloud->points.push_back(center_point_origin);
        }
        return true;
    }

public:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr plane_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr edge_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_z0_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_pub_;

    LidarDetect(rclcpp::Node::SharedPtr node, Params &params)
        : node_(node),
          filtered_cloud_(new pcl::PointCloud<Common::Point>),
          plane_cloud_(new pcl::PointCloud<Common::Point>),
          aligned_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          edge_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          center_z0_cloud_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        x_min_ = params.x_min;
        x_max_ = params.x_max;
        y_min_ = params.y_min;
        y_max_ = params.y_max;
        z_min_ = params.z_min;
        z_max_ = params.z_max;
        circle_radius_ = params.circle_radius;
        delta_width_circles_ = params.delta_width_circles;
        delta_height_circles_ = params.delta_height_circles;
        output_path_ = params.output_path + "/";

        filtered_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("center_cloud", 10);
    }

    // 对齐 ROS1 detect_mech_lidar：按 ring/line 邻域跳变提边缘，再迭代拟合圆 + 几何筛选
    void detect_mech_lidar(pcl::PointCloud<Common::Point>::Ptr cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        filtered_cloud_->clear();
        plane_cloud_->clear();
        aligned_cloud_->clear();
        edge_cloud_->clear();
        center_z0_cloud_->clear();
        center_cloud->clear();

        cropROI(cloud);
        RCLCPP_INFO(node_->get_logger(), "Depth filtered cloud size: %zu", filtered_cloud_->size());

        plane_cloud_->reserve(filtered_cloud_->size());
        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);

        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        RCLCPP_INFO(node_->get_logger(), "Plane cloud size: %zu", plane_cloud_->size());

        edge_cloud_->reserve(filtered_cloud_->size());
        std::unordered_map<unsigned int, std::vector<int>> ring2indices;
        ring2indices.reserve(64);
        for (int i = 0; i < static_cast<int>(filtered_cloud_->size()); ++i)
            ring2indices[filtered_cloud_->points[i].ring].push_back(i);

        const auto &c = plane_coefficients->values;
        Eigen::Vector3d n(c[0], c[1], c[2]);
        double norm_n = n.norm();
        Eigen::Vector3d normal = n / norm_n;

        const double neighbor_gap_threshold = 0.10;
        const int min_points_per_ring = 10;
        for (auto &kv : ring2indices)
        {
            auto &idx_vec = kv.second;
            if (static_cast<int>(idx_vec.size()) < min_points_per_ring)
                continue;
            for (size_t k = 1; k + 1 < idx_vec.size(); ++k)
            {
                const auto &p_prev = filtered_cloud_->points[idx_vec[k - 1]];
                const auto &p_cur = filtered_cloud_->points[idx_vec[k]];
                const auto &p_next = filtered_cloud_->points[idx_vec[k + 1]];

                double dist_plane = std::fabs(c[0] * p_cur.x + c[1] * p_cur.y + c[2] * p_cur.z + c[3]) / norm_n;
                if (dist_plane >= 0.03)
                    continue;

                double dx1 = p_cur.x - p_prev.x, dy1 = p_cur.y - p_prev.y, dz1 = p_cur.z - p_prev.z;
                double dist_prev = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
                double dx2 = p_cur.x - p_next.x, dy2 = p_cur.y - p_next.y, dz2 = p_cur.z - p_next.z;
                double dist_next = std::sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);
                if (dist_prev > neighbor_gap_threshold || dist_next > neighbor_gap_threshold)
                    edge_cloud_->push_back(pcl::PointXYZ(p_cur.x, p_cur.y, p_cur.z));
            }
        }
        RCLCPP_INFO(node_->get_logger(), "Extracted %zu edge points (mechanical LiDAR by neighbor distance).",
                    edge_cloud_->size());
        save2PLY(edge_cloud_, output_path_ + "edge_cloud.ply");

        aligned_cloud_->reserve(edge_cloud_->size());
        Eigen::Matrix3d R_align = rotationAligningNormalToZ(plane_coefficients, normal);
        float average_z = 0.0;
        int cnt = 0;
        for (const auto &pt : *edge_cloud_)
        {
            Eigen::Vector3d aligned_point = R_align * Eigen::Vector3d(pt.x, pt.y, pt.z);
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= std::max(cnt, 1);
        save2PLY(aligned_cloud_, output_path_ + "aligned_cloud.ply");

        pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZ>(*aligned_cloud_));
        fitCirclesIterative(xy_cloud);
        if (center_z0_cloud_->size() < TARGET_NUM_CIRCLES)
        {
            RCLCPP_WARN(node_->get_logger(), "[LiDAR] Only %zu circles found (need 4).", center_z0_cloud_->size());
            return;
        }
        selectCentersByGeometry(R_align, average_z, center_cloud);
    }

    // 对齐 ROS1 detect_solid_lidar 的前半段（体素+平面+PCL 边界），圆拟合改用 ROS1 mech 的迭代 RANSAC。
    // 原因：密集 Livox 边缘点在 cluster_tol=0.05 下会并成 >1000 的大簇，被 MaxClusterSize=1000 丢弃，得到 0 簇。
    void detect_solid_lidar(pcl::PointCloud<Common::Point>::Ptr cloud,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        filtered_cloud_->clear();
        plane_cloud_->clear();
        aligned_cloud_->clear();
        edge_cloud_->clear();
        center_z0_cloud_->clear();
        center_cloud->clear();

        cropROI(cloud);
        RCLCPP_INFO(node_->get_logger(), "Filtered cloud size: %zu", filtered_cloud_->size());

        pcl::VoxelGrid<Common::Point> voxel_filter;
        voxel_filter.setInputCloud(filtered_cloud_);
        voxel_filter.setLeafSize(0.005f, 0.005f, 0.005f);
        voxel_filter.filter(*filtered_cloud_);
        RCLCPP_INFO(node_->get_logger(), "Filtered cloud size: %zu", filtered_cloud_->size());
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*filtered_cloud_, *tmp);
            save2PLY(tmp, output_path_ + "filtered_cloud.ply");
        }

        plane_cloud_->reserve(filtered_cloud_->size());
        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);

        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        RCLCPP_INFO(node_->get_logger(), "Plane cloud size: %zu", plane_cloud_->size());
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*plane_cloud_, *tmp);
            save2PLY(tmp, output_path_ + "plane_cloud.ply");
        }

        aligned_cloud_->reserve(plane_cloud_->size());
        Eigen::Vector3d normal;
        Eigen::Matrix3d R = rotationAligningNormalToZ(plane_coefficients, normal);
        float average_z = 0.0;
        int cnt = 0;
        for (const auto &pt : *plane_cloud_)
        {
            Eigen::Vector3d aligned_point = R * Eigen::Vector3d(pt.x, pt.y, pt.z);
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= std::max(cnt, 1);
        save2PLY(aligned_cloud_, output_path_ + "aligned_cloud.ply");

        // 密集固态雷达：在平面占用栅格上找内部圆孔，再精修圆心
        detectHolesOnAlignedPlane();
        save2PLY(edge_cloud_, output_path_ + "edge_cloud.ply");
        if (center_z0_cloud_->size() < TARGET_NUM_CIRCLES)
        {
            RCLCPP_WARN(node_->get_logger(),
                        "[LiDAR] Occupancy holes=%zu, fallback to iterative RANSAC on plane edges.",
                        center_z0_cloud_->size());
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            normal_estimator.setInputCloud(aligned_cloud_);
            normal_estimator.setRadiusSearch(0.03);
            normal_estimator.compute(*normals);
            pcl::PointCloud<pcl::Boundary> boundaries;
            pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
            boundary_estimator.setInputCloud(aligned_cloud_);
            boundary_estimator.setInputNormals(normals);
            boundary_estimator.setRadiusSearch(0.03);
            boundary_estimator.setAngleThreshold(M_PI / 4);
            boundary_estimator.compute(boundaries);
            edge_cloud_->clear();
            for (size_t i = 0; i < aligned_cloud_->size(); ++i)
            {
                if (boundaries.points[i].boundary_point > 0)
                    edge_cloud_->push_back(aligned_cloud_->points[i]);
            }
            pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZ>(*edge_cloud_));
            fitCirclesIterative(xy_cloud);
        }
        if (center_z0_cloud_->size() < TARGET_NUM_CIRCLES)
        {
            RCLCPP_WARN(node_->get_logger(), "[LiDAR] Only %zu circles found (need 4).", center_z0_cloud_->size());
            return;
        }
        selectCentersByGeometry(R, average_z, center_cloud);
    }

    pcl::PointCloud<Common::Point>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    pcl::PointCloud<Common::Point>::Ptr getPlaneCloud() const { return plane_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif
