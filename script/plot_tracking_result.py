import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from autoware_auto_perception_msgs.msg import ObjectClassification

sys.path.append('../tracking_validation')

# original functions
from utils import *
from tracking_parser import TrackingParser

# this only works in yoshi ri environment
DEFAULT_ROSBAG = "/home/yoshiri/autoware_bag/SPkadai/117/scenario_0_0.db3"
DEFAULT_TOPIC = "/perception/object_recognition/tracking/objects"
DEFAULT_LABEL = "vehicle"

str_labels_map = {
    "vehicle": [ObjectClassification.BUS, ObjectClassification.CAR, ObjectClassification.TRUCK, ObjectClassification.TRAILER],
    "large_vehicle": [ObjectClassification.BUS, ObjectClassification.TRUCK, ObjectClassification.TRAILER],
    "pedestrian": [ObjectClassification.PEDESTRIAN],
    "bike": [ObjectClassification.MOTORCYCLE, ObjectClassification.BICYCLE],
    "all": []
}

def main(args):
    bag_file = args.bag_file
    show = args.show_figure
    tracking_topic = args.topic
    tlim = args.tlim
    xlim = args.xlim
    ylim = args.ylim
    vlim = args.vlim
    yawlim = args.yawlim
    label = args.label
    data_length_lim = args.data_length_lim
    uuid_legend = args.uuid_legend

    tp = TrackingParser(bag_file, tracking_topic)
    if tlim:
        tp.crop_df_by_time("t", tlim[0], tlim[1])
    if xlim:
        tp.filter_df_between("x", xlim[0], xlim[1])
    if ylim:
        tp.filter_df_between("y", ylim[0], ylim[1])
    if vlim:
        tp.filter_df_between("vx", vlim[0], vlim[1])
    if yawlim:
        tp.filter_df_between("yaw", yawlim[0], yawlim[1])
    if label:
        labels = str_labels_map[label]
        tp.filter_df_by_label(labels)

    tp.filter_df_by_data_length(data_length_lim)    
    data = tp.plot_state_and_cov(uuid_legend=uuid_legend)

    tp.plot2d(uuid_legend=uuid_legend)
    if show:
        plt.show()
    return data

def parse_argument():
    parser = argparse.ArgumentParser(description="Example argparse script")

    # bag_file argument
    parser.add_argument(
        "--bag_file",
        type=str,
        default=DEFAULT_ROSBAG,
        help="Path to the rosbag file",
    )

    # show_figure argument
    parser.add_argument(
        "--show_figure",
        type=bool,
        default=True,
        help="Whether to show the figure or not",
    )

    # tracking topic name 
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help="Name of the tracking topic",
    )
    
    # tlim argument
    parser.add_argument(
        "--tlim",
        nargs="+",
        type=float,
        default=[],
        help="relative time limit start with 0 sec. [tmin, tmax]"
    )

    # xlim argument
    parser.add_argument(
        "--xlim",
        nargs="+",
        type=float,
        default=[],
        help="x axis limit [xmin, xmax]"
    )

    # ylim argument
    parser.add_argument(
        "--ylim",
        nargs="+",
        type=float,
        default=[],
        help="y axis limit [ymin, ymax]"
    )

    # vlim argument
    parser.add_argument(
        "--vlim",
        nargs="+",
        type=float,
        default=[],
        help="velocity limit [vmin, vmax]"
    )
    # yawlim argument

    parser.add_argument(
        "--yawlim",
        nargs="+",
        type=float,
        default=[],
        help="yaw limit [yawmin, yawmax]"
    )

    # data length limit argument
    parser.add_argument(
        "--data_length_lim",
        type=int,
        default=0,
        help="data length limit"
    )

    # label
    parser.add_argument(
        "--label",
        type=str,
        default=DEFAULT_LABEL,
        help="Label of the object. Choose from vehicle, pedestrian, bike, all",
    )

    # tracking legend
    parser.add_argument(
        "--uuid_legend",
        type=bool,
        default=False,
        help="Whether to show the legend of tracking as uuid name or not. default is False"
    )

    args = parser.parse_args()
    return args


# show main function
if __name__=="__main__":
    args = parse_argument()
    main(args)