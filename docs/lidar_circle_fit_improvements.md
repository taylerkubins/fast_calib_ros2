# FAST-Calib ROS2 激光圆心拟合改进说明

本文记录将 `FAST-Calib_Ros2` 的 LiDAR 圆孔检测对齐并增强到可用状态的原因、改动与验证结果。

## 1. 问题

ROS2 版本由 ROS1 `FAST-Calib-main` 移植而来，但标定稳定性明显更差，核心表现是 **LiDAR 四个圆孔圆心拟合不准或拟合失败**，后续 SVD 外参也会跟着崩掉。

典型失败日志：

- `Number of edge clusters: 0`：有数千个边缘点，聚类结果却是 0
- `Only 1 circles found (need 4)`：迭代 RANSAC 只找到 1 个圆
- `Skip SVD: qr_centers=4 lidar_centers=0`：相机 4 个圆心正常，激光圆心不足，外参无法计算

在 Mid360 数据（约 2.5 m 距离、ROI 内约 3 万点）上，对齐平面后其实已经能看到 4 个清晰圆孔，问题出在检测算法，而不是点云本身。

## 2. ROS1 / ROS2 原有差异

| 项目 | ROS1 | 旧 ROS2 | 影响 |
|---|---|---|---|
| 雷达分流 | `detect_solid_lidar` / `detect_mech_lidar` | 只有一套聚类拟合 | Livox 与机械雷达无法分路径 |
| 点类型 | `xyz + ring` | 只有 `xyz`，`ring/line` 被丢掉 | 机械雷达邻域跳变提边缘不可用 |
| 固态雷达圆拟合 | PCL 边界 + 欧氏聚类 + 逐簇 CIRCLE2D | 聚类参数被改坏（`tol=0.01`、`min_size=20`、误差阈值 `0.07`） | 圆孔被拆碎或假圆被接受 |
| 机械雷达圆拟合 | 按 ring 邻域跳变提边缘 + **迭代 RANSAC** + 几何筛选 | 无 | 丢掉了 ROS1 更稳的圆拟合 |
| 聚类上限 | `MaxClusterSize=1000` | 同样 1000，且边缘点更密 | 密集点云下整簇被丢弃 |
| 圆心不足仍做 SVD | 提前 return | 继续 SVD | 外参变成乱码 |

ROS1 对 Livox CustomMsg / 无 `ring` 的 PointCloud2 走 **Solid** 路径。Mid360 的 ROS2 PointCloud2 通常只有 `line`、没有 `ring`，因此也会走 Solid。

## 3. 失败原因（针对本次 Mid360 数据）

本次 bag 的中间结果：

- ROI 滤波后：37683 点
- 体素降采样后：33299 点
- 平面内点：27963 点
- PCL 边界点：4707 点

对 `aligned_cloud.ply` 做 1 cm 占用栅格分析，可以得到 4 个内部空洞，圆心约为：

- `(-1.050, -0.188)`
- `(-0.647, -0.186)`
- `(-1.055, 0.308)`
- `(-0.652, 0.309)`

间距约 `0.40 m × 0.50 m`，与标定板 `delta_height_circles / delta_width_circles` 一致。

因此：

1. **直接照搬 ROS1 Solid 的聚类参数会失败**  
   `cluster_tolerance=0.05` 会把圆孔边缘和标定板外框连成超大簇，再被 `MaxClusterSize=1000` 丢掉，得到 0 个簇。

2. **只把 ROS1 Mech 的迭代 RANSAC 套到全部边界点也不够**  
   4707 个边界点混杂了外框、噪声和 4 个圆孔。RANSAC 先拟合出 1 个圆后，剩余点不再满足 `r=0.12±0.03` 的共识，于是只得到 1 个圆。

3. **密集固态雷达更适合“在平面上找空洞”**  
   圆孔是平面内部的空腔，而不是一圈干净的孤立边界。占用栅格比 PCL Boundary + 聚类更稳。

## 4. 代码改进

### 4.1 恢复 ROS1 的雷达分流与点类型

涉及文件：

- `include/common_lib.h`
- `src/data_preprocess.hpp`
- `src/main.cpp`

改动：

- 增加 `Common::Point`（`xyz + ring`）和 `LiDARType::{Solid, Mech}`
- 读 PointCloud2 时保留 `ring`；若无 `ring` 则按 ROS1 规则判为 Solid（Livox 的 `line` 不单独改判为 Mech）
- `main.cpp` 按类型调用 `detect_solid_lidar` / `detect_mech_lidar`
- 圆心不是 4 对时 **跳过 SVD**，避免写出无意义外参

### 4.2 机械雷达路径：完整移植 ROS1 `detect_mech_lidar`

涉及文件：`src/lidar_detect.hpp`

流程：

1. ROI 滤波（不体素降采样）
2. 平面 RANSAC
3. 按 `ring` 分组，用相邻点距离跳变提取孔边缘
4. 对齐到 `Z=0`
5. 在剩余边缘点上迭代 `SACMODEL_CIRCLE2D`（半径限制 `r±0.03`）
6. 用标定板宽高做 `Square` 几何一致性筛选，只保留一组 4 点

### 4.3 固态雷达路径：占用栅格圆孔检测（关键）

涉及文件：`src/lidar_detect.hpp` 中 `detectHolesOnAlignedPlane()`

这是本次能够稳定检出 4 个圆心的核心改动，适用于 Livox Mid360 等密集固态雷达。

流程：

1. ROI 滤波 + 5 mm 体素降采样
2. 平面 RANSAC，将平面旋转对齐到 `Z=0`
3. 在对齐平面上建 **1 cm 占用栅格**
4. 从栅格边界 flood-fill，标记外部背景
5. 剩余封闭空腔即为圆孔候选
6. 用面积过滤：保留接近 \(\pi r^2\) 的连通域（默认 \(0.4\sim 1.8\) 倍）
7. 以空洞质心为初值，用轮廓点 + 圆周附近平面点做 `CIRCLE2D` 精修
8. 再用标定板几何约束选出唯一 4 点，并变换回原始 LiDAR 坐标系

若占用栅格检出不足 4 个，才回退到：PCL 边界提取 + 迭代 RANSAC。

### 4.4 其它配套修改

- `GEOMETRY_TOLERANCE` 与 ROS1 对齐为 `0.08`
- 增加全局 `comb()`，用于从多个候选圆心中枚举 4 点组合
- 迭代 RANSAC 找到一个圆后，会挖掉该圆周带附近的点，避免同一圆被反复拟合
- 调试点云仍输出到 `output_path`：`filtered_cloud.ply`、`plane_cloud.ply`、`aligned_cloud.ply`、`edge_cloud.ply`

## 5. 当前固态雷达检测流程（简图）

```text
bag 点云
  -> ROI 滤波 + Voxel(5mm)
  -> 平面 RANSAC
  -> 旋转对齐到 Z=0
  -> 1cm 占用栅格
  -> flood-fill 找内部空洞
  -> 面积过滤 + CIRCLE2D 精修
  -> Square 几何筛选（0.50 x 0.40）
  -> 逆变换回 LiDAR 系
  -> 与相机 4 圆心 SVD 求 T_cam_lidar
```

## 6. 验证

在数据：

`/home/byd/cxl/bag/0817-hand-hold/rosbag2_2026_08_17-09_43_13`

上，改进后可以稳定检出 4 个 LiDAR 圆心，并完成外参求解。

运行方式：

```bash
source /home/byd/cxl/calib_ws/install/setup.bash
ros2 launch fast_calib calib.launch.py
```

成功时日志应包含类似：

```text
Loaded ... points, lidar_type=Solid
[LiDAR] Hole candidate 1: center=(...) r≈0.12
[LiDAR] Hole candidate 2: ...
[LiDAR] Hole candidate 3: ...
[LiDAR] Hole candidate 4: ...
[Result] RMSE: ... m
```

## 7. 参数说明

标定板几何仍由 `config/qr_params.yaml` 给出，需与实物一致：

- `circle_radius`: 圆孔半径，默认 `0.12`
- `delta_width_circles` / `delta_height_circles`: 圆心间距，默认 `0.50` / `0.40`
- `x_min` ~ `z_max`: 必须把标定板完整包进 ROI，否则空洞不封闭，栅格检测会失败

`circle_fit_error_threshold`、`edge_cluster_tolerance`、`edge_cluster_min_size` 仅为兼容旧聚类路径保留，**当前 Solid 主路径不再使用**。

## 8. 主要改动文件

- `src/FAST-Calib_Ros2/src/lidar_detect.hpp`
- `src/FAST-Calib_Ros2/src/data_preprocess.hpp`
- `src/FAST-Calib_Ros2/src/main.cpp`
- `src/FAST-Calib_Ros2/include/common_lib.h`
- `src/FAST-Calib_Ros2/config/qr_params.yaml`
