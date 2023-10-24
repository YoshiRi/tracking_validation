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
                vel_diff = np.hypot(
                    obj1.kinematics.twist_with_covariance.twist.linear.x - obj2.kinematics.twist_with_covariance.twist.linear.x,
                    obj1.kinematics.twist_with_covariance.twist.linear.y - obj2.kinematics.twist_with_covariance.twist.linear.y
                )
                # print overlap status
                time = detected_objects.header.stamp.sec + detected_objects.header.stamp.nanosec * 1e-9

                if iou == 0:# no overlap
                    results["non_overlap_data"].append([time, distance, yaw_diff, iou, vel_diff])
                    continue 
                # overlap detected
                results["overlap_count"] += 1

                results["overlap_data"].append([time, distance, yaw_diff, iou, vel_diff])
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
    df = pd.DataFrame(data, columns=["time", "distance", "yaw_diff", "iou", "v_diff"])
    # df.plot.line(x="time", y="iou")
    # plt.title("overlapped objects iou")
    df.plot.scatter(x="distance", y="yaw_diff")
    plt.title("overlapped objects distance vs yaw_diff")
    plt.grid()
    restrict_max_plot_range(20, 1.54) # 20m, 1.54rad
    if save:
        plt.savefig(save_path + "overlap_distance_yaw_diff.png")

    df.plot.scatter(x="distance", y="v_diff")
    plt.title("overlapped objects distance vs v_diff")
    plt.grid()
    restrict_max_plot_range(20, 10) # 20m, 10m/s
    if save:
        plt.savefig(save_path + "overlap_distance_v_diff.png")

    data = results["non_overlap_data"]
    df = pd.DataFrame(data, columns=["time", "distance", "yaw_diff", "iou", "v_diff"])
    df.plot.scatter(x="distance", y="yaw_diff", color="orange")
    plt.title("non overlapped objects distance vs yaw_diff")
    plt.grid()
    restrict_max_plot_range(20, 1.54) # 20m, 1.54rad
    if save:
        plt.savefig(save_path + "non_overlap_distance_yaw_diff.png")


def analyze_rosbag(bagfile, topic, save=False, save_path="./"):
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



def run_with_multiple_bag_files(arg):
    import glob
    path = arg.save_path
    # bagfiles = glob.glob("/home/yoshiri/Downloads/temp/xx1_kashiwa_radar/*/*")
    bagfiles = glob.glob("/home/yoshiri/Downloads/temp/radar*/tracking_result/*.db3")
    path = "odaiba_1/"
    print(bagfiles)

    total_results = {}
    total_results["overlap_data"] = []
    total_results["non_overlap_data"] = []
    total_results["overlap_count"] = 0
    for bagfile in bagfiles:
        print(bagfile)
        results = analyze_rosbag(bagfile, arg.topic, save=True, save_path=path)
        # merge results
        total_results["overlap_data"].extend(results["overlap_data"])
        total_results["non_overlap_data"].extend(results["non_overlap_data"])
        total_results["overlap_count"] += results["overlap_count"]
    
    # visualize total results
    plot_overlap_results(total_results, save=True, save_path=path+"total_results_") 
    print("total overlap count: ", total_results["overlap_count"])



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

    # save path argument
    parser.add_argument(
        "--save_path",
        type=str,
        default="images/",
        help="save path",
    )
    
    # multiple bag files flag
    parser.add_argument(
        "--multiple_bag_files",
        type=bool,
        default=False,
        help="whether to use multiple bag files or not",
    )
    
    # save flag
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="whether to save or not",
    )

    arg = parser.parse_args()
    
    # normal process
    if not arg.multiple_bag_files:
        analyze_rosbag(arg.bag_file, arg.topic, save=arg.save, save_path=arg.save_path)
    else:
        run_with_multiple_bag_files(arg)


