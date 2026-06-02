#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将相机图像颜色投影到 LiDAR 点云，生成带颜色的点云，用于验证 FAST-Calib_Ros2 标定得到的外参。
支持从 FAST-Calib_Ros2 的 calib_result.txt 读取内参与外参（LiDAR→相机）。

用法（ROS2 工作空间下）:
  # 默认从 output/lidar_pcd_output 取第一个 PCD（LiDAR 系）+ output 下图像，会用 R、T 变换
  python3 project_image_to_pointcloud.py --calib /path/to/output/calib_result.txt

  # 使用自己的点云/图像路径
  python3 project_image_to_pointcloud.py --calib /path/to/calib_result.txt --pcd /path/to/lidar.pcd --image /path/to/image.png --output colored.ply
"""
import argparse
import re
import os
import numpy as np
import cv2
import json
import open3d as o3d


def load_calib_result_txt(txt_path: str):
    """
    从 FAST-Calib_Ros2 输出的 calib_result.txt 读取内参 K 和外参 R、T（LiDAR→相机）。
    返回 K (3x3), R (3x3), T (3,)。
    """
    with open(txt_path, "r") as f:
        content = f.read()
    fx = float(re.search(r"cam_fx:\s*([\d.+-]+)", content).group(1))
    fy = float(re.search(r"cam_fy:\s*([\d.+-]+)", content).group(1))
    cx = float(re.search(r"cam_cx:\s*([\d.+-]+)", content).group(1))
    cy = float(re.search(r"cam_cy:\s*([\d.+-]+)", content).group(1))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    # Rcl/Pcl 中数字用逗号分隔，需在字符类中包含逗号
    rcl = re.search(r"Rcl:\s*\[\s*([\d.\s+,\-]+)\]", content, re.DOTALL)
    pcl = re.search(r"Pcl:\s*\[\s*([\d.\s+,\-]+)\]", content)
    if not rcl or not pcl:
        raise ValueError("calib_result.txt 中未找到 Rcl 或 Pcl")
    rnums = [float(x) for x in re.findall(r"-?[\d.]+", rcl.group(1))]
    pnums = [float(x) for x in re.findall(r"-?[\d.]+", pcl.group(1))]
    R = np.array(rnums, dtype=np.float64).reshape(3, 3)
    T = np.array(pnums, dtype=np.float64)
    return K, R, T


def find_first_pcd_in_dir(dir_path: str) -> str:
    """在目录中取第一个 .pcd 文件（按文件名排序），用于 lidar_pcd_output。"""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    pcds = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".pcd")])
    if not pcds:
        raise FileNotFoundError(f"目录中未找到 .pcd 文件: {dir_path}")
    return os.path.join(dir_path, pcds[0])


def project_image_color_to_pointcloud(
    pcd_path: str,
    img_path: str,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    output_path: str = "colored_pointcloud.ply",
    use_nearest_pixel: bool = True,
    pcd_in_camera_frame: bool = False,
):
    """
    将图像颜色投影到点云并保存带颜色的点云

    参数:
        pcd_path: 输入点云路径（支持 .pcd/.ply 等格式）
        img_path: 输入图像路径（支持 .jpg/.png 等格式）
        K: 相机内参矩阵（3x3），格式为 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        R: 雷达到相机的旋转矩阵（3x3）
        T: 雷达到相机的平移向量（3x1 或 3 元素数组，单位：米）
        output_path: 输出带颜色点云的保存路径（推荐 .ply 格式）
        use_nearest_pixel: 是否使用最近邻像素颜色（True）或双线性插值（False）
        pcd_in_camera_frame: 若为 True，认为点云已在相机坐标系，不再施加 R、T（用于 FAST-Calib 输出的 colored_cloud.pcd）
    """
    # ---------------------- 步骤1：读取点云和图像 ----------------------
    pcd = o3d.io.read_point_cloud(pcd_path)
    points_input = np.asarray(pcd.points)  # 输入点云（Nx3），可能是雷达系或相机系

    if len(points_input) == 0:
        raise ValueError(f"点云文件为空或无法读取: {pcd_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"未找到图像文件: {img_path}")
    img_height, img_width = img.shape[:2]

    # ---------------------- 步骤2：得到相机坐标系点 ----------------------
    R = R.reshape(3, 3)
    T = T.reshape(3, 1) if T.ndim == 1 else T

    if pcd_in_camera_frame:
        points_cam = points_input.copy()
    else:
        # 雷达点云 → 相机坐标系（公式：P_cam = R @ P_lidar + T）
        points_cam = (R @ points_input.T).T + T.squeeze()


    # ---------------------- 步骤3：投影到图像平面并过滤无效点 ----------------------
    valid_z_mask = points_cam[:, 2] > 0
    n_valid_z = int(np.sum(valid_z_mask))
    points_cam_valid = points_cam[valid_z_mask]
    points_output_valid = points_input[valid_z_mask]  # 与 points_cam 同序，用于最终输出

    # 投影公式：u = fx*(X/Z) + cx；v = fy*(Y/Z) + cy
    X, Y, Z = points_cam_valid[:, 0], points_cam_valid[:, 1], points_cam_valid[:, 2]
    u = (K[0, 0] * X) / Z + K[0, 2]  # 等价于 (fx*X + cx*Z)/Z
    v = (K[1, 1] * Y) / Z + K[1, 2]  # 等价于 (fy*Y + cy*Z)/Z

    # 转换为像素坐标并过滤超出图像边界的点
    if use_nearest_pixel:
        u_int = np.round(u).astype(int)  # 最近邻取整
        v_int = np.round(v).astype(int)
    else:
        u_int = np.floor(u).astype(int)  # 双线性插值需保留浮点坐标（后续处理）
        v_int = np.floor(v).astype(int)

    in_image_mask = (u_int >= 0) & (u_int < img_width) & (v_int >= 0) & (v_int < img_height)
    n_in_image = int(np.sum(in_image_mask))
    u_valid = u_int[in_image_mask]
    v_valid = v_int[in_image_mask]
    points_to_save = points_output_valid[in_image_mask]

    print(f"[诊断] 输入点数: {len(points_input)}, 相机前方(z>0): {n_valid_z}, 落在图像内: {n_in_image}")

    if len(points_to_save) == 0:
        print(
            "[错误] 没有点通过过滤（相机前方且落在图像内）。\n"
            "若使用的是 FAST-Calib 输出的 colored_cloud.pcd，请加上参数: --pcd-in-camera-frame\n"
            "否则请确认点云与图像来自同一帧，且标定结果与当前相机/雷达一致。"
        )
        return None

    # ---------------------- 步骤4：获取对应像素的颜色 ----------------------
    colors = np.zeros((len(points_to_save), 3), dtype=np.float32)

    if use_nearest_pixel:
        # 直接取最近邻像素的颜色
        for i, (u, v) in enumerate(zip(u_valid, v_valid)):
            b, g, r = img[v, u]  # OpenCV读取的是BGR格式
            colors[i] = [r/255.0, g/255.0, b/255.0]  # 转换为RGB并归一化到0-1
    else:
        # 双线性插值（适用于点云投影点不在像素中心的情况）
        u_frac = u[in_image_mask] - u_valid  # 小数部分
        v_frac = v[in_image_mask] - v_valid
        for i in range(len(u_valid)):
            u0, v0 = u_valid[i], v_valid[i]
            u1, v1 = u0 + 1, v0 + 1

            # 边界处理（防止越界）
            u0 = np.clip(u0, 0, img_width-1)
            u1 = np.clip(u1, 0, img_width-1)
            v0 = np.clip(v0, 0, img_height-1)
            v1 = np.clip(v1, 0, img_height-1)

            # 四个邻域像素的颜色
            color00 = img[v0, u0][::-1]  # BGR→RGB
            color01 = img[v0, u1][::-1]
            color10 = img[v1, u0][::-1]
            color11 = img[v1, u1][::-1]

            # 双线性插值计算
            color = (
                (1 - u_frac[i]) * (1 - v_frac[i]) * color00 +
                u_frac[i] * (1 - v_frac[i]) * color01 +
                (1 - u_frac[i]) * v_frac[i] * color10 +
                u_frac[i] * v_frac[i] * color11
            )
            colors[i] = color / 255.0  # 归一化到0-1


    # ---------------------- 步骤5：创建带颜色的点云并保存 ----------------------
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points_to_save)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, colored_pcd, write_ascii=True, compressed=False, print_progress=True)
    ratio = len(points_to_save) / len(points_input)
    print(f"带颜色的点云已保存至: {output_path}（原始点保留率：{ratio:.2%}）")

    return colored_pcd


def calculate_final_extrinsic_matrix(R0, T0, R1, T1):
    """
    计算原始雷达数据到相机的最终外参矩阵

    参数:
    R0 (np.ndarray): 点云预处理旋转矩阵 (3x3)
    T0 (np.ndarray): 点云预处理平移向量 (3,)
    R1 (np.ndarray): 预处理后的雷达到相机旋转矩阵 (3x3)
    T1 (np.ndarray): 预处理后的雷达到相机平移向量 (3,)

    返回:
    tuple: (R, T)，其中 R 是最终旋转矩阵 (3x3)，T 是最终平移向量 (3,)
    """
    # 计算最终旋转矩阵: R = R1 * R0
    R = np.dot(R1, R0)

    # 计算最终平移向量: T = R1 * T0 + T1
    T = np.dot(R1, T0) + T1

    return R, T


def save_RT_to_json(R, T, out_json_path):
    with open(out_json_path, 'w') as f:
        json_data = {
            "R": [
                [R[0, 0], R[0, 1], R[0, 2]],
                [R[1, 0], R[1, 1], R[1, 2]],
                [R[2, 0], R[2, 1], R[2, 2]]
            ],
            "T": T[:].tolist()
        }
        json.dump(json_data, f, indent=4)
        print(f"标定结果已保存至: {out_json_path}")
    return


def load_camera_intrinsic(json_file):
    """
    加载相机内参和外参（替换为你的实际标定结果）
    """

    with open(json_file, 'r') as f:
        data = json.load(f)
        _K = np.array(data["K"], dtype=np.float64)
        _D = np.array(data["D"], dtype=np.float64)
        print(f"loading camera intrinsic params:\nK:{_K}\nD:{_D}\n")

    return _K, _D


def load_extrinsic(json_file):
    """从 JSON 加载外参 R、T（兼容旧用法）。"""
    with open(json_file, "r") as f:
        data = json.load(f)
    _R = np.array(data["R"], dtype=np.float64)
    _T = np.array(data["T"], dtype=np.float64)
    print(f"loading extrinsic params:\nR:{_R}\nT:{_T}\n")
    return _R, _T


def main():
    parser = argparse.ArgumentParser(
        description="将相机图像颜色投影到 LiDAR 点云，验证标定外参（适配 FAST-Calib_Ros2 输出）"
    )
    parser.add_argument("--calib", type=str, default="", help="FAST-Calib_Ros2 的 calib_result.txt 路径（推荐）")
    parser.add_argument("--intrinsic", type=str, default="", help="相机内参 JSON（与 --extrinsic 一起使用）")
    parser.add_argument("--extrinsic", type=str, default="", help="外参 JSON（与 --intrinsic 一起使用）")
    parser.add_argument("--pcd", type=str, default="", help="点云文件路径（.pcd/.ply）")
    parser.add_argument("--image", type=str, default="", help="图像文件路径")
    parser.add_argument("--output", type=str, default="colored_pointcloud.ply", help="输出带颜色点云路径（推荐 .ply）")
    parser.add_argument("--pcd-in-camera-frame", action="store_true", help="点云已在相机坐标系（FAST-Calib 的 colored_cloud.pcd 即为此种）")
    parser.add_argument("--no-nearest", action="store_true", help="使用双线性插值取色（默认使用最近邻）")
    parser.add_argument("--vis", action="store_true", help="投影完成后用 Open3D 可视化")
    parser.add_argument("--vis-downsample", type=int, default=0, metavar="N", help="可视化前每 N 点取 1（如 4 可减轻 Jetson 显存/NvMap 错误）")
    args = parser.parse_args()

    if args.calib:
        calib_path = os.path.abspath(args.calib) if not os.path.isabs(args.calib) else args.calib
        K, R, T = load_calib_result_txt(calib_path)
        out_dir = os.path.dirname(calib_path)
        pcd_dir = os.path.join(out_dir, "lidar_pcd_output")
        try:
            default_pcd = find_first_pcd_in_dir(pcd_dir)
        except FileNotFoundError as e:
            print(e)
            return 1
        default_img = os.path.join(out_dir, "qr_detect.png")
    else:
        if not args.intrinsic or not args.extrinsic:
            print("请指定 --calib 或同时指定 --intrinsic 与 --extrinsic")
            return 1
        K, _ = load_camera_intrinsic(args.intrinsic)
        R, T = load_extrinsic(args.extrinsic)
        default_pcd = ""
        default_img = ""

    pcd_path = args.pcd or default_pcd
    img_path = args.image or default_img
    if not pcd_path or not img_path:
        print("请指定 --pcd 和 --image，或使用 --calib 指向 output 目录下的 calib_result.txt（点云从 output/lidar_pcd_output 取第一个 .pcd）")
        return 1

    # 默认从 lidar_pcd_output 读取的 PCD 为 LiDAR 系，需用 R、T 变换到相机系
    if args.calib and not args.pcd:
        pcd_in_camera_frame = False
    else:
        pcd_in_camera_frame = args.pcd_in_camera_frame

    colored_pcd = project_image_color_to_pointcloud(
        pcd_path=pcd_path,
        img_path=img_path,
        K=K,
        R=R,
        T=T,
        output_path=args.output,
        use_nearest_pixel=not args.no_nearest,
        pcd_in_camera_frame=pcd_in_camera_frame,
    )
    if colored_pcd is None:
        return 1
    if args.vis:
        vis_pcd = colored_pcd
        if args.vis_downsample > 1:
            vis_pcd = colored_pcd.uniform_down_sample(args.vis_downsample)
            print(f"[可视化] 下采样后点数: {len(vis_pcd.points)}（原始 {len(colored_pcd.points)}）")
        print("[提示] 若出现 NvMap 错误，可尝试: --vis-downsample 4  或  LIBGL_ALWAYS_SOFTWARE=1 python3 ...")
        o3d.visualization.draw_geometries([vis_pcd])
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
