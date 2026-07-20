import os
from glob import glob

from setuptools import setup


package_name = "autodrive_f1tenth"


setup(
    name="autodrive-f1tenth",
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="malik",
    maintainer_email="malik@example.com",
    description="Consolidated AutoDRIVE F1TENTH ROS 2 package with simulator bridges and tracking/control stack.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "autodrive_incoming_bridge = autodrive_f1tenth.autodrive_incoming_bridge:main",
            "autodrive_outgoing_bridge = autodrive_f1tenth.autodrive_outgoing_bridge:main",
            "autodrive_incoming_bridge_2 = autodrive_f1tenth.autodrive_incoming_bridge_2:main",
            "autodrive_outgoing_bridge_2 = autodrive_f1tenth.autodrive_outgoing_bridge_2:main",
            "teleop_keyboard = autodrive_f1tenth.teleop_keyboard:main",
            "pure_pursuit = autodrive_f1tenth.pure_pursuit:main",
            "gap_follow = autodrive_f1tenth.gap_follow:main",
            "pure_pursuit_target = autodrive_f1tenth.pure_pursuit_target:main",
            "confidence_log = autodrive_f1tenth.confidence_log:main",
            "safety = autodrive_f1tenth.safety:main",
            "slam = autodrive_f1tenth.slam:main",
            "slam_tf_logger = autodrive_f1tenth.slam_tf_logger:main",
            "slam_toolbox_map_logger = autodrive_f1tenth.slam_toolbox_map_logger:main",
            "slam_toolbox_bridge = autodrive_f1tenth.slam_toolbox_bridge:main",
            "dl_slot = autodrive_f1tenth.dl_slot:main",
            "dl_slot_scan = autodrive_f1tenth.dl_slot_scan:main",
            "target_vehicle_tracker = autodrive_f1tenth.target_vehicle_tracker:main",
            "target_vehicle_tracker_foreground = autodrive_f1tenth.target_vehicle_tracker_foreground:main",
            "overtake_decision = autodrive_f1tenth.overtake_decision:main",
            "follow_stack = autodrive_f1tenth.follow_stack:main",
            "plot_pure_pursuit_runs = autodrive_f1tenth.plot_pure_pursuit_runs:main",
            "thesis = autodrive_f1tenth.thesis:main",
        ],
    },
)
