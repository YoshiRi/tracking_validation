import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
sys.path.append('../tracking_validation')

# original functions
from utils import *
from tracking_parser import TrackingParser

# this only works in yoshi ri environment
DEFAULT_ROSBAG = "/home/yoshiri/autoware_bag/SPkadai/117/scenario_0_0.db3"
DEFAULT_TOPIC = "/perception/object_recognition/tracking/objects"

def main(bag_file, show = False, tracking_topic = DEFAULT_TOPIC, xlim = [], ylim = []):
    tp = TrackingParser(bag_file, tracking_topic)
    if xlim:
        tp.filter_df_between("x", xlim[0], xlim[1])
    if ylim:
        tp.filter_df_between("y", ylim[0], ylim[1])
    data = tp.plot_data()
    if show:
        plt.show()
    return data


if __name__=="__main__":
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
        "--tracking_topic",
        type=str,
        default=DEFAULT_TOPIC,
        help="Name of the tracking topic",
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

    args = parser.parse_args()

    main(args.bag_file, args.show_figure, args.tracking_topic, args.xlim, args.ylim)