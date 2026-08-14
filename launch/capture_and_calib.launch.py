from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Whether to launch RViz for debug visualization.",
    )
    lidar_topic_arg = DeclareLaunchArgument(
        "lidar_topic",
        default_value="/rslidar_points",
        description="LiDAR PointCloud2 topic to record.",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/color/image_raw",
        description="Camera image topic used for one-shot capture.",
    )
    capture_duration_arg = DeclareLaunchArgument(
        "capture_duration",
        default_value="15.0",
        description="Seconds to record LiDAR bag.",
    )
    snapshot_timeout_arg = DeclareLaunchArgument(
        "snapshot_timeout",
        default_value="12.0",
        description="Seconds to wait for a camera image.",
    )
    output_root_arg = DeclareLaunchArgument(
        "output_root",
        default_value="/tmp/fast_calib_capture",
        description="Directory where temporary bag/image are stored.",
    )
    session_name_arg = DeclareLaunchArgument(
        "session_name",
        default_value="run",
        description="Subfolder name under output_root.",
    )

    pkg_share = FindPackageShare("fast_calib")
    params_file = PathJoinSubstitution([pkg_share, "config", "qr_params.yaml"])
    capture_script = PathJoinSubstitution([pkg_share, "..", "..", "lib", "fast_calib", "capture_one_image.py"])

    session_dir = PathJoinSubstitution([LaunchConfiguration("output_root"), LaunchConfiguration("session_name")])
    bag_dir = PathJoinSubstitution([session_dir, "bag"])
    image_path = PathJoinSubstitution([session_dir, "snapshot.png"])

    record_lidar = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-d",
            LaunchConfiguration("capture_duration"),
            "-o",
            bag_dir,
            LaunchConfiguration("lidar_topic"),
        ],
        output="screen",
    )

    capture_one_image = ExecuteProcess(
        cmd=[
            capture_script,
            "--topic",
            LaunchConfiguration("image_topic"),
            "--output",
            image_path,
            "--timeout",
            LaunchConfiguration("snapshot_timeout"),
        ],
        output="screen",
    )

    fast_calib_node = Node(
        package="fast_calib",
        executable="fast_calib",
        name="fast_calib",
        output="screen",
        parameters=[
            params_file,
            {
                "bag_path": bag_dir,
                "image_path": image_path,
                "lidar_topic": LaunchConfiguration("lidar_topic"),
            },
        ],
    )

    # Start calibration shortly after recording duration to ensure bag is closed.
    delayed_fast_calib = TimerAction(
        period=PythonExpression([LaunchConfiguration("capture_duration"), " + 1.0"]),
        actions=[fast_calib_node],
    )

    rviz_config = PathJoinSubstitution([pkg_share, "rviz_cfg", "fast_livo2.rviz"])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            rviz_arg,
            lidar_topic_arg,
            image_topic_arg,
            capture_duration_arg,
            snapshot_timeout_arg,
            output_root_arg,
            session_name_arg,
            record_lidar,
            capture_one_image,
            delayed_fast_calib,
            rviz_node,
        ]
    )
