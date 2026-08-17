#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ROS2 的 .db3 bag 中按间隔帧提取多张图像到指定目录。

用法示例：
  python3 extract_images_from_db3.py --db3 /path/to/rosbag2_xxx_0.db3 --output ./images --interval 10
  python3 extract_images_from_db3.py -d /path/to/bag_dir -o ./out -i 5 --topic /camera/image_raw

参数：
  --db3 / -d     .db3 文件路径或 bag 所在目录（目录需含 metadata.yaml 与 .db3）
  --output / -o  输出目录，默认 ./extracted_images
  --interval / -i  每间隔几帧保存一张图，1=每帧都保存，10=每 10 帧保存 1 张，默认 1
  --topic / -t   指定图像话题（不指定则自动选第一个 Image/CompressedImage 话题）
  --prefix       输出文件名前缀，默认 image，得到 image_000000.png, image_000001.png ...
  --format       输出格式：png 或 jpg，默认 png

依赖：rosbags、opencv-python、numpy（无需 source ROS2）
  pip3 install rosbags opencv-python numpy
"""

import argparse
import os
import sys
import numpy as np

try:
    import cv2
except ImportError:
    print("请安装 opencv-python: pip3 install opencv-python", file=sys.stderr)
    sys.exit(1)

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("请安装 rosbags: pip3 install rosbags", file=sys.stderr)
    sys.exit(1)


# 支持的图像消息类型
IMAGE_MSG_TYPES = (
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
)


def get_bag_directory(db3_path):
    """若传入的是 .db3 文件路径，返回其所在目录；否则返回路径本身（视为 bag 目录）。"""
    path = os.path.abspath(db3_path)
    if os.path.isfile(path) and path.lower().endswith(".db3"):
        return os.path.dirname(path)
    return path


def image_message_to_cv2(typestore, msg_type_name, rawdata):
    """
    将 ROS2 Image / CompressedImage 反序列化并转为 OpenCV BGR 图像 (numpy ndarray)。
    返回 (img_array, None) 成功，或 (None, error_msg) 失败。
    """
    try:
        msg = typestore.deserialize_cdr(rawdata, msg_type_name)
    except Exception as e:
        return None, str(e)

    if msg_type_name == "sensor_msgs/msg/CompressedImage":
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None, "cv2.imdecode 失败"
            return img, None
        except Exception as e:
            return None, str(e)

    if msg_type_name == "sensor_msgs/msg/Image":
        try:
            h, w = msg.height, msg.width
            step = msg.step if msg.step else w * 3
            enc = (msg.encoding or "bgr8").strip().lower()
            ch = 3 if ("rgb" in enc or "bgr" in enc) else 1
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, step))
            arr = arr[:, : w * ch]
            if ch == 1:
                img = arr
            else:
                img = arr.reshape((h, w, ch))
                if "rgb" in enc:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img, None
        except Exception as e:
            return None, str(e)

    return None, f"不支持的图像类型: {msg_type_name}"


def extract_images(db3_path, output_dir, interval=1, topic=None, prefix="image", fmt="png"):
    """
    从 db3/bag 目录中按间隔提取图像。

    :param db3_path: .db3 文件路径或 bag 目录
    :param output_dir: 输出目录
    :param interval: 每 interval 帧保存 1 张（1=每帧都保存）
    :param topic: 图像话题名，None 表示自动选择
    :param prefix: 输出文件名前缀
    :param fmt: 扩展名，png 或 jpg
    """
    bag_dir = get_bag_directory(db3_path)
    if not os.path.isdir(bag_dir):
        print(f"错误：bag 目录不存在: {bag_dir}", file=sys.stderr)
        return
    if not os.path.exists(os.path.join(bag_dir, "metadata.yaml")):
        print(f"警告：未在 {bag_dir} 下找到 metadata.yaml，仍尝试读取。", file=sys.stderr)

    os.makedirs(output_dir, exist_ok=True)
    ext = "png" if fmt.lower() in ("png", "jpg", "jpeg") else fmt.lower()
    if ext != "png":
        ext = "jpg"

    try:
        typestore = get_typestore(Stores.ROS2_HUMBLE)
    except Exception:
        typestore = get_typestore(Stores.LATEST)

    with Reader(bag_dir) as reader:
        topics = reader.topics
        image_topics = [
            t for t, info in topics.items()
            if info.msgtype in IMAGE_MSG_TYPES
        ]
        if not image_topics:
            print("错误：未找到 Image 或 CompressedImage 类型的话题。", file=sys.stderr)
            print("可用话题及类型:", file=sys.stderr)
            for t, info in topics.items():
                print(f"  {t}  ->  {info.msgtype}", file=sys.stderr)
            return

        if topic is not None:
            if topic not in image_topics:
                print(f"错误：指定话题 '{topic}' 不是图像类型或不存在。", file=sys.stderr)
                print("图像话题示例:", image_topics, file=sys.stderr)
                return
            target_topic = topic
        else:
            target_topic = image_topics[0]
            if len(image_topics) > 1:
                print(f"未指定 --topic，使用第一个图像话题: {target_topic}")
                print(f"其他图像话题: {[t for t in image_topics if t != target_topic]}")

        msg_type = topics[target_topic].msgtype
        conns = [c for c in reader.connections if c.topic == target_topic]
        if not conns:
            print("错误：未找到该话题的 connection。", file=sys.stderr)
            return
        frame_index = 0
        saved_count = 0

        for connection, timestamp, rawdata in reader.messages(connections=conns):
            if frame_index % interval != 0:
                frame_index += 1
                continue

            img, err = image_message_to_cv2(typestore, connection.msgtype, rawdata)
            if err or img is None:
                print(f"警告：帧 {frame_index} 解码失败: {err}", file=sys.stderr)
                frame_index += 1
                continue

            out_name = f"{prefix}_{saved_count:06d}.{ext}"
            out_path = os.path.join(output_dir, out_name)
            if cv2.imwrite(out_path, img):
                saved_count += 1
                if saved_count <= 3 or saved_count % 50 == 0:
                    print(f"已保存: {out_name}")
            else:
                print(f"警告：写入失败 {out_path}", file=sys.stderr)
            frame_index += 1

    print(f"\n完成。共保存 {saved_count} 张图像到: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="从 .db3 / rosbag2 中按间隔提取多张图像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-d", "--db3", required=True, help=".db3 文件路径或 bag 目录")
    parser.add_argument("-o", "--output", default="./extracted_images", help="输出目录（默认: ./extracted_images）")
    parser.add_argument("-i", "--interval", type=int, default=1, help="每间隔几帧保存一张（默认: 1，即每帧都保存）")
    parser.add_argument("-t", "--topic", default=None, help="图像话题（不指定则自动选第一个图像话题）")
    parser.add_argument("--prefix", default="image", help="输出文件名前缀（默认: image）")
    parser.add_argument("--format", choices=("png", "jpg"), default="png", help="输出格式（默认: png）")

    args = parser.parse_args()
    if args.interval < 1:
        args.interval = 1

    extract_images(
        db3_path=args.db3,
        output_dir=args.output,
        interval=args.interval,
        topic=args.topic,
        prefix=args.prefix,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
