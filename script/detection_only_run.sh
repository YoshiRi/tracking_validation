# load args
rosbag_file=$1
# default start_offset is 0.01
start_offset=${2:-0.01}
# default speed is 0.2
speed=${3:-0.2}

# launch rosbag play
ros2 bag play $rosbag_file -r $speed -s sqlite3 --start-offset $start_offset --remap /perception/object_recognition/tracking/objects:=/tracking /perception/object_recognition/objects:=/prediction --clock