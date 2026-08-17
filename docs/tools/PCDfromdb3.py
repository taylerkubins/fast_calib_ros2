#!/usr/bin/env python3
"""
从 .db3 格式 ROS2 bag 提取激光雷达点云为 PCD 文件
支持话题：sensor_msgs/PointCloud2 格式的激光雷达数据
输出：按时间戳命名的 PCD 文件，兼容 CloudCompare/PCL 打开

依赖：rosbags（新版 API 使用 typestore.deserialize_cdr，不再从 serde 导入 deserialize_cdr）
"""
import argparse
import os
import numpy as np
import open3d as o3d
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

def get_point_cloud_data(msg):
    """解析 PointCloud2 消息为 xyz 点云数据"""
    # 解析点云字段（找到 x/y/z 对应的偏移量）
    x_offset = y_offset = z_offset = None
    for field in msg.fields:
        if field.name == 'x':
            x_offset = field.offset
        elif field.name == 'y':
            y_offset = field.offset
        elif field.name == 'z':
            z_offset = field.offset
    
    # 兼容无字段信息的情况（默认 float32 格式，x/y/z 依次排列）
    if None in [x_offset, y_offset, z_offset]:
        # 假设点云是 float32 (x,y,z) 或 (x,y,z,intensity) 格式
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]  # 只取 x,y,z
    
    # 按字段偏移量解析（兼容自定义点云格式）
    points = []
    for i in range(0, len(msg.data), msg.point_step):
        # 提取 x/y/z（float32 类型）
        x = np.frombuffer(msg.data[i+x_offset:i+x_offset+4], dtype=np.float32)[0]
        y = np.frombuffer(msg.data[i+y_offset:i+y_offset+4], dtype=np.float32)[0]
        z = np.frombuffer(msg.data[i+z_offset:i+z_offset+4], dtype=np.float32)[0]
        points.append([x, y, z])
    
    return np.array(points, dtype=np.float32)

def db3_to_pcd(bag_path, topic_name, output_dir, skip_frames=0):
    """
    从.db3 bag包提取PCD文件
    :param bag_path: .db3文件路径
    :param topic_name: 激光雷达话题名（如/points_raw）
    :param output_dir: PCD输出目录
    :param skip_frames: 跳过帧数（减少输出文件数量，0=保存所有帧）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.exists(bag_path):
        print(f"错误：找不到文件 {bag_path}")
        return
    
    # 使用内置 ROS2 类型库反序列化（新版 rosbags 无 serde.deserialize_cdr，改用 typestore）
    try:
        typestore = get_typestore(Stores.LATEST)
    except Exception:
        typestore = get_typestore(Stores.ROS2_HUMBLE)

    with Reader(bag_path) as reader:
        topic_list = list(reader.topics.keys())
        if topic_name not in topic_list:
            print(f"错误：话题 {topic_name} 不存在！")
            print(f"可用话题列表：{topic_list}")
            return

        frame_count = 0
        saved_count = 0
        print(f"开始提取 {bag_path} 中 {topic_name} 的点云...")

        for connection, timestamp, rawdata in reader.messages():
            if connection.topic != topic_name:
                continue
            if frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            try:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            except Exception as e:
                print(f"警告：反序列化第 {frame_count} 帧失败: {e}")
                frame_count += 1
                continue

            try:
                points = get_point_cloud_data(msg)
                if len(points) == 0:
                    print(f"警告：第 {frame_count} 帧点云为空，跳过")
                    frame_count += 1
                    continue
            except Exception as e:
                print(f"警告：解析第 {frame_count} 帧失败: {e}")
                frame_count += 1
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            timestamp_str = str(timestamp)[:10]
            pcd_filename = os.path.join(output_dir, f"lidar_{timestamp_str}_{saved_count:06d}.pcd")
            o3d.io.write_point_cloud(pcd_filename, pcd)
            print(f"已保存：{os.path.basename(pcd_filename)} (共 {len(points)} 个点)")
            frame_count += 1
            saved_count += 1
    
    print(f"\n提取完成！共保存 {saved_count} 个PCD文件到 {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从.db3格式ROS2 bag提取激光雷达PCD文件')
    parser.add_argument('--db3', required=True, help='.db3文件路径（如 rosbag2_xxx_0.db3）')
    parser.add_argument('--topic', required=True, help='激光雷达话题名（如 /points_raw、/lidar_points）')
    parser.add_argument('--output', default='./lidar_pcd_output', help='PCD输出目录（默认：./lidar_pcd_output）')
    parser.add_argument('--skip', type=int, default=0, help='跳过帧数（如 --skip 9 表示每10帧保存1帧，减少文件数）')
    
    args = parser.parse_args()
    
    # 自动安装依赖
    try:
        import open3d
        from rosbags.rosbag2 import Reader
    except ImportError:
        print("正在安装依赖包...")
        os.system("pip3 install rosbags open3d numpy --upgrade")
        import open3d
        from rosbags.rosbag2 import Reader
    
    # 执行提取
    db3_to_pcd(args.db3, args.topic, args.output, args.skip)

