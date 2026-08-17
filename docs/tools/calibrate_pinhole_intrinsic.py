#
import cv2
import os
import glob
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm


def calibrate_camera(images_folder, pattern_size, square_size):
    """相机标定函数，返回内参、畸变系数和外参"""
    # 准备棋盘格角点的世界坐标
    # 创建一个 (pattern_size[0] * pattern_size[1], 3) 的零矩阵，用于存储棋盘格角点的3D坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    # 使用 np.mgrid 生成棋盘格角点的二维坐标，然后重塑并赋值给 objp 的前两列（Z坐标保持为0）
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    # 将坐标乘以实际的方格尺寸，得到真实的物理坐标
    objp *= square_size

    # 存储所有图像的对象点和图像点
    obj_points = []  # 3D世界坐标
    img_points = []  # 2D图像坐标
    img_shapes = []  # 图像尺寸
    images_used = []  # 成功处理的图像路径

    # 获取所有棋盘格图像
    # 首先尝试查找所有 .jpg 格式的图像
    images = glob.glob(os.path.join(images_folder, "*.jpg"))
    # 如果没有找到 .jpg 图像，则尝试查找 .png 格式
    if not images:
        images = glob.glob(os.path.join(images_folder, "*.png"))
    # 如果仍然没有找到图像，则抛出异常
    if not images:
        raise ValueError("No images found in the specified folder")

    print(f"找到 {len(images)} 张图像，开始处理...")

    # 检测角点
    # 使用 tqdm 创建进度条，遍历所有图像
    for fname in tqdm(images, desc="检测棋盘格角点"):
        # 读取图像
        img = cv2.imread(fname)
        # 如果图像读取失败，则跳过
        if img is None:
            continue

        # 将图像转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 存储图像尺寸 (width, height)
        img_shapes.append(gray.shape[::-1])  # 存储图像尺寸 (w, h)

        # 查找棋盘格角点
        # 使用 cv2.findChessboardCorners 查找角点
        # ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # 可以尝试添加更多标志位来提高角点检测精度
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)

        # 如果成功找到角点
        if ret:
            # 亚像素精确化
            # 设置角点检测的终止条件
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # 使用 cv2.cornerSubPix 对角点进行亚像素精确化
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 将世界坐标和图像坐标分别添加到对应的列表中
            obj_points.append(objp)
            img_points.append(corners2)
            images_used.append(fname)

    # 如果成功检测到的图像少于5张，则抛出异常
    if len(obj_points) < 5:
        raise ValueError("至少需要5张成功检测棋盘格的图像进行标定")

    print(f"成功检测到 {len(obj_points)} 张图像的棋盘格角点")

    # 相机标定
    print("正在进行相机标定...")
    # 使用 cv2.calibrateCamera 进行相机标定
    # 参数说明：
    # obj_points: 世界坐标点
    # img_points: 图像坐标点
    # img_shapes[0]: 图像尺寸
    # None, None: 初始内参矩阵和畸变系数（这里设为None，让函数自动计算）
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shapes[0], None, None)

    # 返回标定结果
    # ret: 重投影误差
    # camera_matrix: 内参矩阵
    # dist_coeffs: 畸变系数
    # rvecs: 旋转向量列表
    # tvecs: 平移向量列表
    # obj_points: 世界坐标点列表
    # images_used: 成功处理的图像路径列表
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs, obj_points, images_used


def generate_camera_centric_point_cloud(camera_matrix, rvecs, tvecs, obj_points):
    """生成以相机光心为原点的点云"""
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    all_points = []
    colors = []

    # 添加相机光心（红色，位于原点）
    all_points.append([0, 0, 0])  # 相机光心在相机坐标系中始终为(0,0,0)
    colors.append([1, 0, 0])  # 红色表示相机光心

    # 添加棋盘格角点（蓝色）
    for i in range(len(rvecs)):
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvecs[i])

        # 将棋盘格点转换到相机坐标系
        # 公式: P_camera = R * P_world + t
        for obj_point in obj_points[i]:
            # 将点从世界坐标系转换到相机坐标系
            camera_coord = np.dot(R, obj_point) + tvecs[i].reshape(3)
            all_points.append(camera_coord)
            colors.append([0, 0, 1])  # 蓝色表示棋盘格点

    # 转换为numpy数组
    points_array = np.array(all_points, dtype=np.float32)
    colors_array = np.array(colors, dtype=np.float32)

    # 设置点云数据
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.colors = o3d.utility.Vector3dVector(colors_array)

    return pcd


def save_point_cloud_with_camera_frustums(pcd, camera_matrix, rvecs, tvecs, output_file, scale=0.8):
    """保存点云并添加相机视锥体"""
    # 创建坐标轴（表示相机坐标系）
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # 创建相机视锥体
    camera_frustums = []
    for i in range(len(rvecs)):
        # 创建相机视锥体
        frustum = create_camera_frustum(camera_matrix, rvecs[i], tvecs[i], scale=scale)
        camera_frustums.append(frustum)

    # 保存所有几何体
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"点云已保存至: {output_file}")

    # 可视化
    geometries = [pcd, coordinate_frame] + camera_frustums
    o3d.visualization.draw_geometries(geometries, window_name="相机坐标系中的点云与视锥体")


def create_camera_frustum(camera_matrix, rvec, tvec, scale=0.8):
    """创建正确的相机视锥体，以光心为中心，方向与相机坐标系一致"""
    # 获取相机内参
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # 假设图像尺寸（可以从标定中获取或估计）
    image_width = 2 * cx  # 基于像主点估计图像宽度
    image_height = 2 * cy  # 基于像主点估计图像高度

    # 计算视野角度（FOV）
    fov_x = 2 * np.arctan2(image_width, 2 * fx)
    fov_y = 2 * np.arctan2(image_height, 2 * fy)

    # 计算视锥体尺寸（基于比例尺）
    near = scale * 0.5  # 近平面距离
    far = scale * 2.0  # 远平面距离

    # 计算近平面尺寸
    near_height = 2 * np.tan(fov_y / 2) * near
    near_width = 2 * np.tan(fov_x / 2) * near

    # 计算远平面尺寸
    far_height = 2 * np.tan(fov_y / 2) * far
    far_width = 2 * np.tan(fov_x / 2) * far

    # 在相机坐标系中定义视锥体顶点
    # 顶点顺序：光心、近平面四个角、远平面四个角
    points = np.array([
        # 光心 (0)
        [0, 0, 0],

        # 近平面四个角 (1-4)
        [-near_width / 2, -near_height / 2, near],  # 左下
        [near_width / 2, -near_height / 2, near],  # 右下
        [near_width / 2, near_height / 2, near],  # 右上
        [-near_width / 2, near_height / 2, near],  # 左上

        # 远平面四个角 (5-8)
        [-far_width / 2, -far_height / 2, far],  # 左下
        [far_width / 2, -far_height / 2, far],  # 右下
        [far_width / 2, far_height / 2, far],  # 右上
        [-far_width / 2, far_height / 2, far]  # 左上
    ], dtype=np.float32)

    # 定义连接线（光心到近平面、近平面到远平面、远平面轮廓）
    lines = [
        # 光心到近平面
        [0, 1], [0, 2], [0, 3], [0, 4],

        # 光心到远平面（可选）
        [0, 5], [0, 6], [0, 7], [0, 8],

        # 近平面轮廓
        [1, 2], [2, 3], [3, 4], [4, 1],

        # 远平面轮廓
        [5, 6], [6, 7], [7, 8], [8, 5],

        # 连接近远平面
        [1, 5], [2, 6], [3, 7], [4, 8]
    ]

    # 创建线集
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # 设置颜色（绿色表示视锥体）
    colors = [[0, 1, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def save_intrinsic_to_json(out_json_path, mtx, dist, ret):
    """
    将相机内参和畸变系数保存到JSON文件中
    
    参数:
    out_json_path: 输出JSON文件的路径
    mtx: 相机内参矩阵 (3x3)
    dist: 畸变系数矩阵 (1x5)
    """
    # 以写入模式打开JSON文件
    with open(out_json_path, 'w') as f:
        # 构造要保存的JSON数据字典
        json_data = {
            # 相机内参矩阵 K (3x3矩阵)
            "K": [
                [mtx[0, 0], mtx[0, 1], mtx[0, 2]],  # 第一行: [fx, skew, cx]
                [mtx[1, 0], mtx[1, 1], mtx[1, 2]],  # 第二行: [0, fy, cy]
                [mtx[2, 0], mtx[2, 1], mtx[2, 2]]   # 第三行: [0, 0, 1]
            ],
            # 畸变系数向量 D (5个参数)
            "D": dist[0, :].tolist(),  # 提取第一行的所有列并转换为列表

            # 单独保存内参矩阵中的关键参数
            "fx": mtx[0, 0],  # x轴焦距(像素)
            "fy": mtx[1, 1],  # y轴焦距(像素)
            "cx": mtx[0, 2],  # 图像主点x坐标(像素)
            "cy": mtx[1, 2],  # 图像主点y坐标(像素)
            # 单独保存畸变系数中的各个参数
            "k1": dist[0, 0],  # 径向畸变系数1
            "k2": dist[0, 1],  # 径向畸变系数2
            "d1": dist[0, 2],  # 切向畸变系数1
            "d2": dist[0, 3],  # 切向畸变系数2
            "k3": dist[0, 4],  # 径向畸变系数3
            "reproj_error": ret,  # 重投影误差
        }
        # 将数据以JSON格式写入文件，缩进4个空格以提高可读性
        json.dump(json_data, f, indent=4)
        # 打印保存成功的消息
        print(f"标定结果已保存至: {out_json_path}")
    return

def compute_pose_for_new_image(new_image_path, camera_matrix, dist_coeffs, 
                              pattern_size, square_size):
    """
    为新图像计算位姿
    """
    # 读取并处理图像
    img = cv2.imread(new_image_path)
    if img is None:
        print(f"无法加载图像: {new_image_path}")
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 准备世界坐标点
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 检测角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # 亚像素精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 计算外参
        success, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
        
        if success:
            print(f"成功计算图像 {new_image_path} 的外参")
            return rvec, tvec
        else:
            print("solvePnP计算失败")
            return None, None
    else:
        print(f"在图像 {new_image_path} 中未检测到棋盘格")
        return None, None
    

# 参数设置
images_folder = r"/home/jetson/Desktop/xiangji_2026_03_13-09_52_25/out"  # 替换为你的图像文件夹路径
pattern_size = (11, 8)  # 棋盘格w内部角点数 (宽, 高)
square_size = 0.035  # 棋盘格方块尺寸 (单位：米)
output_pcd = os.path.join(images_folder, "camera_centric_points.ply")
out_json_path = os.path.join(images_folder, "intrinsic.json")

# 步骤1: 相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs, obj_points, used_images = calibrate_camera(
    images_folder, pattern_size, square_size)
save_intrinsic_to_json(out_json_path, camera_matrix, dist_coeffs, ret)

print("\n相机标定完成!")
print(f"内参矩阵:\n{camera_matrix}")
print(f"畸变系数: {dist_coeffs.flatten()}")
print(f"重投影误差: {ret:.4f} 像素")

# 步骤2: 生成以相机光心为原点的点云
pcd = generate_camera_centric_point_cloud(camera_matrix, rvecs, tvecs, obj_points)

# 步骤3: 保存点云并添加相机视锥体
save_point_cloud_with_camera_frustums(pcd, camera_matrix, rvecs, tvecs, output_pcd)

# 打印相机光心位置（始终为原点）
print("\n相机光心在相机坐标系中的坐标 (固定为原点):")
print(f"X: 0.0000 m")
print(f"Y: 0.0000 m")
print(f"Z: 0.0000 m")

# 打印棋盘格点统计信息
points = np.asarray(pcd.points)
chessboard_points = points[1:]  # 排除第一个点（相机光心）
print(f"\n棋盘格点统计:")
print(f"点数: {len(chessboard_points)}")
print(f"X范围: [{np.min(chessboard_points[:, 0]):.4f}, {np.max(chessboard_points[:, 0]):.4f}] m")
print(f"Y范围: [{np.min(chessboard_points[:, 1]):.4f}, {np.max(chessboard_points[:, 1]):.4f}] m")
print(f"Z范围: [{np.min(chessboard_points[:, 2]):.4f}, {np.max(chessboard_points[:, 2]):.4f}] m")

# 为新图像计算外参
# new_image_path = "/home/biaoding/bd_dataset/cam extr/image_182.png"
# rvec_new, tvec_new = compute_pose_for_new_image(
#     new_image_path, camera_matrix, dist_coeffs, pattern_size, square_size)

# if rvec_new is not None and tvec_new is not None:
#     R_new, _ = cv2.Rodrigues(rvec_new)
#     print("新图像的外参:")
#     print(f"旋转矩阵:\n{R_new}")
#     print(f"平移向量:\n{tvec_new.flatten()}")
