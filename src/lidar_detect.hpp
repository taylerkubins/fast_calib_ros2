/*
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIDAR_DETECT_HPP
#define LIDAR_DETECT_HPP
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>

#include <algorithm>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/ply_io.h>
#include "common_lib.h"

class LidarDetect
{
private:
    // Axis-aligned ROI limits used to crop the incoming cloud before processing.
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    // Expected hole radius on the calibration target (in meters).
    double circle_radius_;
    // 圆拟合可接受误差阈值，平均径向误差小于该值的圆心才被接受（由配置 circle_fit_error_threshold 设定）
    double circle_fit_error_threshold_;
    // 两圆心平面距离小于此值视为同一圆，合并时保留误差更小者
    double circle_center_merge_distance_;
    // 边缘聚类最小点数
    int edge_cluster_min_size_;
    rclcpp::Node::SharedPtr node_;
    std::string output_path_;

    // Cached intermediate clouds so they can be visualized in RViz/debug outputs.
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_z0_cloud_;

public:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr plane_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr edge_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_z0_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_pub_;

    LidarDetect(rclcpp::Node::SharedPtr node, Params &params)
        : node_(node),
          filtered_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          plane_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
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
        circle_fit_error_threshold_ = params.circle_fit_error_threshold;
        circle_center_merge_distance_ = params.circle_center_merge_distance;
        edge_cluster_min_size_ = params.edge_cluster_min_size;
        output_path_ = params.output_path + "/";

        filtered_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("center_cloud", 10);
    }

    void detect_lidar(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // Pipeline summary:
        // 1) Crop cloud by ROI and downsample.
        // 2) Extract target plane with RANSAC.
        // 3) Rotate plane to a canonical Z=0 frame.
        // 4) Detect boundary (edge) points on that plane.
        // 5) Cluster edges into candidate circular rims.
        // 6) Fit circles to each cluster and keep valid centers.

        // 1) Pass-through filtering in X/Y/Z keeps only the region where
        // the calibration target is expected.
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);
        pass_x.filter(*filtered_cloud_);

        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);
        pass_y.filter(*filtered_cloud_);

        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);
        pass_z.filter(*filtered_cloud_);

        RCLCPP_INFO(node_->get_logger(), "Filtered cloud size: %ld", filtered_cloud_->size());

        // Voxel downsampling reduces point density/noise and speeds up
        // all following steps.
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(filtered_cloud_);
        voxel_filter.setLeafSize(0.005f, 0.005f, 0.005f);
        voxel_filter.filter(*filtered_cloud_);
        RCLCPP_INFO(node_->get_logger(), "Filtered cloud size: %ld", filtered_cloud_->size());
        save2PLY(filtered_cloud_, output_path_ + "filtered_cloud.ply");
        // 2) Plane segmentation: find dominant board plane via RANSAC.
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        // Max point-to-plane distance to count as inlier.
        plane_segmentation.setDistanceThreshold(0.01);   //0.01);
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        RCLCPP_INFO(node_->get_logger(), "Plane cloud size: %ld", plane_cloud_->size());
        save2PLY(plane_cloud_, output_path_ + "plane_cloud.ply");
        // 3) Align the extracted plane with Z=0 so circle fitting is easier
        // and can be done in 2D.
        aligned_cloud_->reserve(plane_cloud_->size());

        Eigen::Vector3d normal(plane_coefficients->values[0],
                               plane_coefficients->values[1],
                               plane_coefficients->values[2]);
        normal.normalize();
        Eigen::Vector3d z_axis(0, 0, 1);

        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(normal.dot(z_axis));

        Eigen::AngleAxisd rotation(angle, axis);
        Eigen::Matrix3d R = rotation.toRotationMatrix();

        // Rotate each plane point by R. We intentionally store z=0 in the
        // aligned cloud and keep the average original z separately.
        float average_z = 0.0;
        int cnt = 0;
        for (const auto &pt : *plane_cloud_)
        {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= cnt;
        save2PLY(aligned_cloud_, output_path_ + "aligned_cloud.ply");
        // 4) Edge extraction:
        // Estimate local normals, then mark boundary points.
        edge_cloud_->reserve(aligned_cloud_->size());

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimator.setInputCloud(aligned_cloud_);
        // Neighborhood radius used for normal estimation.
        normal_estimator.setRadiusSearch(0.03);
        normal_estimator.compute(*normals);

        pcl::PointCloud<pcl::Boundary> boundaries;
        pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
        boundary_estimator.setInputCloud(aligned_cloud_);
        boundary_estimator.setInputNormals(normals);
        // Radius and angular threshold controlling boundary sensitivity.
        boundary_estimator.setRadiusSearch(0.03);
        boundary_estimator.setAngleThreshold(M_PI / 4);
        boundary_estimator.compute(boundaries);

        for (size_t i = 0; i < aligned_cloud_->size(); ++i)
        {
            if (boundaries.points[i].boundary_point > 0)
            {
                edge_cloud_->push_back(aligned_cloud_->points[i]);
            }
        }
        RCLCPP_INFO(node_->get_logger(), "Extracted %ld edge points.", edge_cloud_->size());
        save2PLY(edge_cloud_, output_path_ + "edge_cloud.ply");
        // 5) Cluster edge points so each cluster can correspond to one
        // physical contour (hole rim, board edge, or outlier structure).
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(edge_cloud_);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        // NOTE: Tolerance/min/max size are critical tuning parameters.
        // Too loose merges different contours; too strict fragments circles.
        ec.setClusterTolerance(0.01); // 0.02
        ec.setMinClusterSize(edge_cluster_min_size_);
        ec.setMaxClusterSize(1000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(edge_cloud_);
        ec.extract(cluster_indices);

        RCLCPP_INFO(node_->get_logger(), "Number of edge clusters: %ld", cluster_indices.size());
        RCLCPP_INFO(node_->get_logger(), "Cluster Tolerance: %f", ec.getClusterTolerance());


        // 6) Circle fitting per cluster (in aligned Z=0 frame).
        center_z0_cloud_->clear();
        center_cloud->points.clear();
        Eigen::Matrix3d R_inv = R.inverse();

        // 先收集所有误差 < 0.02 的候选圆心 (error, center_origin, center_z0)，最后若多于 4 个则按误差取最好的 4 个
        std::vector<std::pair<double, std::pair<pcl::PointXYZ, pcl::PointXYZ>>> candidates;

        for (size_t i = 0; i < cluster_indices.size(); ++i)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto &idx : cluster_indices[i].indices)
            {
                cluster->push_back(edge_cloud_->points[idx]);
            }

            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_CIRCLE2D);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01);
            seg.setMaxIterations(1000);
            seg.setInputCloud(cluster);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() > 0)
            {
                double error = 0.0;
                for (const auto &idx : inliers->indices)
                {
                    double dx = cluster->points[idx].x - coefficients->values[0];
                    double dy = cluster->points[idx].y - coefficients->values[1];
                    double distance = sqrt(dx * dx + dy * dy) - circle_radius_;
                    error += abs(distance);
                }
                error /= inliers->indices.size();

                RCLCPP_DEBUG(node_->get_logger(), "cluster %zu inliers %zu error %.6f", i, inliers->indices.size(), error);
                if (error < circle_fit_error_threshold_)
                {
                    pcl::PointXYZ center_point;
                    center_point.x = coefficients->values[0];
                    center_point.y = coefficients->values[1];
                    center_point.z = 0.0;
                    Eigen::Vector3d aligned_point(center_point.x, center_point.y, center_point.z + average_z);
                    Eigen::Vector3d original_point = R_inv * aligned_point;
                    pcl::PointXYZ center_point_origin;
                    center_point_origin.x = original_point.x();
                    center_point_origin.y = original_point.y();
                    center_point_origin.z = original_point.z();
                    candidates.push_back({ error, { center_point_origin, center_point } });
                }
            }
        }

        // 合并近重复圆心：同一圆可能被多个边缘簇拟合出多个中心，平面距离小于 merge_distance 的只保留误差最小者
        std::sort(candidates.begin(), candidates.end(),
                  [](const std::pair<double, std::pair<pcl::PointXYZ, pcl::PointXYZ>> &a,
                     const std::pair<double, std::pair<pcl::PointXYZ, pcl::PointXYZ>> &b) { return a.first < b.first; });
        std::vector<std::pair<double, std::pair<pcl::PointXYZ, pcl::PointXYZ>>> merged;
        const double md2 = circle_center_merge_distance_ * circle_center_merge_distance_;
        for (const auto &c : candidates)
        {
            const pcl::PointXYZ &p0 = c.second.second; // center in aligned Z=0 frame
            bool skip = false;
            for (const auto &m : merged)
            {
                const pcl::PointXYZ &p1 = m.second.second;
                double dx = p0.x - p1.x, dy = p0.y - p1.y;
                if (dx * dx + dy * dy < md2)
                {
                    skip = true;
                    break;
                }
            }
            if (!skip)
                merged.push_back(c);
        }
        if (merged.size() < candidates.size())
            RCLCPP_INFO(node_->get_logger(), "LiDAR: merged %zu duplicate circle centers -> %zu unique.", candidates.size(), merged.size());
        candidates = std::move(merged);

        // 若候选多于 4 个，只保留误差最小的 4 个（标定板 4 个圆）
        const size_t n_keep = 4;
        const size_t n_candidates = candidates.size();
        if (n_candidates > n_keep)
        {
            std::sort(candidates.begin(), candidates.end(),
                      [](const std::pair<double, std::pair<pcl::PointXYZ, pcl::PointXYZ>> &a,
                         const std::pair<double, std::pair<pcl::PointXYZ, pcl::PointXYZ>> &b) { return a.first < b.first; });
            candidates.resize(n_keep);
            RCLCPP_WARN(node_->get_logger(), "LiDAR: %zu circle candidates found, keeping 4 with smallest fit error.", n_candidates);
        }
        for (const auto &c : candidates)
        {
            center_cloud->points.push_back(c.second.first);
            center_z0_cloud_->push_back(c.second.second);
        }
        if (center_cloud->size() != 4)
        {
            RCLCPP_WARN(node_->get_logger(), "LiDAR: detected %zu circle centers (need 4). Check ROI and circle_radius in config.", center_cloud->size());
        }
    }

    // Accessors for RViz/debug publishing from main.cpp.
    pcl::PointCloud<pcl::PointXYZ>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getPlaneCloud() const { return plane_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif
