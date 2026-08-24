import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/malik/ros2_ws/src/autodrive_ros2/autodrive_f1tenth/install/autodrive_f1tenth'
