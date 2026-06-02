# colored_cloud.ply 独立生成工具

从 ROS2 bag、相机图像、LiDAR→Camera 外参与相机内参，生成与 FAST-Calib `colored_cloud.ply` 逻辑一致的彩色点云。

## 依赖

```bash
pip install -r requirements.txt
```

## 配置

编辑 `config/reprojection_params.yaml`：

| 参数 | 说明 |
|------|------|
| `bag_path` | ROS2 bag 目录或 `.db3` 文件 |
| `lidar_topic` | LiDAR 点云话题 |
| `image_topic` | 相机图像话题（留空则自动选 bag 中第一个 Image 话题） |
| `image_frame_index` | 使用 bag 中第几帧图像（0 = 第一帧） |
| `extrinsic_calib_path` | FAST-LIVO2 格式 `calib_result.txt`（含 Rcl/Pcl） |
| `output_path` | 输出目录 |
| `camera` | 内参（若 calib_result 中已有则以其为准，YAML 可覆盖） |

## 运行

```bash
cd /home/byd/cxl/calib_ws/reprojection_ws
python3 generate_colored_cloud.py --config config/reprojection_params.yaml
```

输出：

- `output_path/colored_cloud.ply`
- `output_path/colored_cloud.pcd`（可选，`save_pcd: true`）

## 算法说明

与 `FAST-Calib_Ros2/include/common_lib.h::projectPointCloudToImage` 一致：

1. 从 bag 读取第一帧（或指定帧）相机图像
2. 读取 bag 中全部 LiDAR 帧并合并（同 `data_preprocess.hpp`）
3. `cv2.undistort` 去畸变图像
4. 点云经 `T_cam_lidar` 变换到相机系，丢弃 `Z < 0`
5. `cv2.projectPoints` 投影（零畸变）
6. 输出点坐标为**相机系** XYZ，颜色来自 undistort 图像 BGR→RGB

## Python API

```python
from colored_cloud_generator import generate_colored_cloud

result = generate_colored_cloud("config/reprojection_params.yaml")
print(result["ply_path"])
```
