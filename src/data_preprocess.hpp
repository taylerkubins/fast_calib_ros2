/*
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef DATA_PREPROCESS_HPP
#define DATA_PREPROCESS_HPP

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cstring>
#include <iostream>

#include "common_lib.h"

using namespace std;
using namespace cv;

class DataPreprocess
{
public:
    pcl::PointCloud<Common::Point>::Ptr cloud_input_;
    cv::Mat img_input_;
    LiDARType lidar_type_{LiDARType::Unknown};

    DataPreprocess(Params &params)
        : cloud_input_(new pcl::PointCloud<Common::Point>)
    {
        string bag_path = params.bag_path;
        string image_path = params.image_path;
        string lidar_topic = params.lidar_topic;

        img_input_ = cv::imread(params.image_path, cv::IMREAD_UNCHANGED);
        if (img_input_.empty())
        {
            std::cerr << "Loading the image " << image_path << " failed" << std::endl;
            return;
        }
        std::cout << "Successfully loaded image: " << image_path << std::endl;

        std::cout << "Attempting to load point cloud from ROS2 bag: " << bag_path << std::endl;
        std::cout << "Looking for topic: " << lidar_topic << std::endl;

        loadPointCloudFromBag(bag_path, lidar_topic, cloud_input_);
        std::cout << "Loaded " << cloud_input_->size() << " points, lidar_type="
                  << (lidar_type_ == LiDARType::Mech ? "Mech" :
                      lidar_type_ == LiDARType::Solid ? "Solid" : "Unknown")
                  << std::endl;
    }

private:
    static const sensor_msgs::msg::PointField *findField(const sensor_msgs::msg::PointCloud2 &msg,
                                                         const std::string &name)
    {
        for (const auto &f : msg.fields)
        {
            if (f.name == name)
                return &f;
        }
        return nullptr;
    }

    static uint16_t readUIntField(const sensor_msgs::msg::PointCloud2 &msg,
                                  const sensor_msgs::msg::PointField &field,
                                  size_t point_index)
    {
        const uint8_t *ptr = msg.data.data() + point_index * msg.point_step + field.offset;
        uint16_t v16 = 0;
        uint8_t v8 = 0;
        int32_t v32 = 0;
        switch (field.datatype)
        {
        case sensor_msgs::msg::PointField::UINT8:
            return *ptr;
        case sensor_msgs::msg::PointField::INT8:
            return static_cast<uint16_t>(*reinterpret_cast<const int8_t *>(ptr));
        case sensor_msgs::msg::PointField::UINT16:
            std::memcpy(&v16, ptr, sizeof(v16));
            return v16;
        case sensor_msgs::msg::PointField::INT32:
            std::memcpy(&v32, ptr, sizeof(v32));
            return static_cast<uint16_t>(v32);
        default:
            return 0;
        }
    }

    bool loadPointCloudFromBag(
        const std::string &bag_path,
        const std::string &topic_name,
        pcl::PointCloud<Common::Point>::Ptr &cloud)
    {
        try
        {
            rosbag2_cpp::Reader reader;
            reader.open(bag_path);
            rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serialization;

            while (reader.has_next())
            {
                auto bag_message = reader.read_next();
                if (bag_message->topic_name != topic_name)
                    continue;

                auto ros_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
                rclcpp::SerializedMessage extracted_serialized_msg(*bag_message->serialized_data);
                serialization.deserialize_message(&extracted_serialized_msg, ros_msg.get());

                const auto *ring_field = findField(*ros_msg, "ring");
                const auto *line_field = findField(*ros_msg, "line");
                // 与 ROS1 一致：只有 ring 才走机械雷达路径；Livox 的 line 仍按固态处理
                lidar_type_ = (ring_field != nullptr) ? LiDARType::Mech : LiDARType::Solid;

                sensor_msgs::PointCloud2ConstIterator<float> it_x(*ros_msg, "x");
                sensor_msgs::PointCloud2ConstIterator<float> it_y(*ros_msg, "y");
                sensor_msgs::PointCloud2ConstIterator<float> it_z(*ros_msg, "z");

                const size_t n = static_cast<size_t>(ros_msg->width) * ros_msg->height;
                cloud->reserve(cloud->size() + n);
                for (size_t i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z)
                {
                    Common::Point p;
                    p.x = *it_x;
                    p.y = *it_y;
                    p.z = *it_z;
                    if (ring_field)
                        p.ring = readUIntField(*ros_msg, *ring_field, i);
                    else if (line_field)
                        p.ring = readUIntField(*ros_msg, *line_field, i);
                    else
                        p.ring = 0xFFFF;
                    cloud->push_back(p);
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error reading bag file: " << e.what() << std::endl;
            return false;
        }
        return true;
    }
};

typedef std::shared_ptr<DataPreprocess> DataPreprocessPtr;

#endif
