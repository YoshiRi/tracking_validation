import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from autoware_auto_perception_msgs.msg import ObjectClassification, DetectedObject, DetectedObjects, TrackedObject, TrackedObjects
from typing import List, Dict, Tuple, Union, Optional

Objects = Union[DetectedObjects, TrackedObjects]
Object = Union[DetectedObject, TrackedObject]

sys.path.append('../tracking_validation')

DEFAULT_ROSBAG = "/home/yoshiri/Downloads/temp/xx1_kashiwa_radar/14-38/211130ae-8a64-4311-95ab-1b8406c4499b_2023-10-18-14-38-32_p0900_0.db3"
RADAR_TOPIC = "/perception/object_recognition/detection/radar/far_objects"

# original functions
from utils import *
from tracking_parser import DetectionParser
from iou_utils import *

# create 

# 望ましくないobjectのoverlapを定義する関数
# def define_object_overlap(obj1: DetectedObject, obj2: DetectedObject):
    # BBOXの重なり率
    


# listを受け取って時刻ごとのObjectのoverlapがないか確認する関数
# objectのoverlapをどのように定義するか
def check_radar_detection_overlap(detections: list[Objects]):
    results = {}
    results["overlap_data"] = []
    results["non_overlap_data"] = []
    results["overlap_count"] = 0    
    for detected_objects in detections:
        objects: list[DetectedObject] = detected_objects.objects
        if len(objects) < 2:
            continue
        
        # get all pairs of objects
        for i in range(len(objects)):
            for j in range(i+1, len(objects)):
                obj1 = objects[i]
                obj2 = objects[j]
                # check overlap by iou
                iou = get_2d_iou(obj1, obj2)
                distance = np.hypot(
                    obj1.kinematics.pose_with_covariance.pose.position.x - obj2.kinematics.pose_with_covariance.pose.position.x,
                    obj1.kinematics.pose_with_covariance.pose.position.y - obj2.kinematics.pose_with_covariance.pose.position.y
                )
                yaw_diff = OrientationToYaw(obj1.kinematics.pose_with_covariance.pose.orientation) - OrientationToYaw(obj2.kinematics.pose_with_covariance.pose.orientation)
                # normalize radians
                yaw_diff = abs(np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff)))
                # print overlap status
                time = detected_objects.header.stamp.sec + detected_objects.header.stamp.nanosec * 1e-9

                if iou == 0:# no overlap
                    results["non_overlap_data"].append([time, distance, yaw_diff, iou])
                    continue 
                # overlap detected
                results["overlap_count"] += 1

                results["overlap_data"].append([time, distance, yaw_diff, iou])
    print("overlap count: ", results["overlap_count"])
    return results

def restrict_max_plot_range(max_x, max_y):
    # get current axis limits
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # set new axes limits
    new_max_x = min(xlim[1], max_x)
    new_max_y = min(ylim[1], max_y)
    ax.set_xlim([xlim[0], new_max_x])
    ax.set_ylim([ylim[0], new_max_y])

def plot_overlap_results(results, save=False, save_path=""):
    data = results["overlap_data"]
    df = pd.DataFrame(data, columns=["time", "distance", "yaw_diff", "iou"])
    # df.plot.line(x="time", y="iou")
    # plt.title("overlapped objects iou")
    df.plot.scatter(x="distance", y="yaw_diff")
    plt.title("overlapped objects distance vs yaw_diff")
    plt.grid()
    restrict_max_plot_range(20, 1.54) # 20m, 1.54rad
    if save:
        plt.savefig(save_path + "overlap_distance_yaw_diff.png")

    data = results["non_overlap_data"]
    df = pd.DataFrame(data, columns=["time", "distance", "yaw_diff", "iou"])
    df.plot.scatter(x="distance", y="yaw_diff", color="orange")
    plt.title("non overlapped objects distance vs yaw_diff")
    plt.grid()
    restrict_max_plot_range(20, 1.54) # 20m, 1.54rad
    if save:
        plt.savefig(save_path + "non_overlap_distance_yaw_diff.png")


def main(bagfile, topic, save=False, save_path="./"):
    dp = DetectionParser(bagfile, RADAR_TOPIC)
    list_detections = dp.msg_list

    results = check_radar_detection_overlap(list_detections)
    if save:
        bagfile_name = bagfile.split("/")[-1]
        save_file_path = save_path + bagfile_name.split(".")[0] + "_"
        plot_overlap_results(results, save=True, save_path=save_file_path)
    else:
        plot_overlap_results(results)
        plt.show()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example argparse script")

    # bag_file argument
    parser.add_argument(
        "--bag_file",
        type=str,
        default=DEFAULT_ROSBAG,
        help="Path to the rosbag file",
    )
    # topic argument
    parser.add_argument(
        "--topic",
        type=str,
        default=RADAR_TOPIC,
        help="topic name",
    )

    arg = parser.parse_args()
    
    # normal process
    main(arg.bag_file, arg.topic, save=False, save_path="images/")