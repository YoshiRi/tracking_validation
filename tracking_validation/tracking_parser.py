from rosbag_parser import get_topics_as_dict, get_topics_and_tf
import uuid
from utils import *
import matplotlib.pyplot as plt
import numpy as np


def getUUID(obj):
    return str(uuid.UUID(bytes=obj.object_id.uuid.tobytes()))

def getTrackersInID(tracker_topic_dict):
    trackers = {}
    # get trackers
    for stamp, msg in tracker_topic_dict:
        time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for tracked_obj in msg.objects:
            id = getUUID(tracked_obj)
            if id not in trackers.keys():
                trackers[id] = []
            trackers[id].append([time, tracked_obj])
    return trackers

def getTrackingData(bagfile, topic_name = "/perception/object_recognition/tracking/objects"):
    topic_dict, tf_buffer = get_topics_and_tf(bagfile, [topic_name])[topic_name]
    trackers = getTrackersInID(topic_dict)
    return trackers


def getTrackerKinematicsList(trackers: list):
    data = []
    for time, topic in trackers:
        xy = get2DPosition(topic)
        yaw = getYaw(topic)
        vx = getVX(topic)
        length = topic.shape.dimensions.x
        width = topic.shape.dimensions.y
        data.append([time, xy[0], xy[1], yaw, vx,length, width])
    
    data = np.array(data).reshape(-1,7)
    return data

def getTrackerKinematicsDict(trackers: list):
    data = {}
    keys = ["time", "x", "y", "yaw", "vx", "length", "width",
             "covariance_x", "covariance_y", "covariance_xy_matrix",
             "covariance_vx", "covariance_yaw"]
    for key in keys:
        data[key] = []

    for time, topic in trackers:
        xy = get2DPosition(topic)
        data["time"].append(time)
        data["x"].append(xy[0])
        data["y"].append(xy[1])
        data["yaw"].append(getYaw(topic))
        data["vx"].append(getVX(topic))
        data["length"].append(topic.shape.dimensions.x)
        data["width"].append(topic.shape.dimensions.y)
        cov2d = get2DPositionCovariance(topic)
        data["covariance_x"].append(cov2d[0])
        data["covariance_y"].append(cov2d[1])
        data["covariance_yaw"].append(cov2d[2])
        data["covariance_xy_matrix"].append(get2DPositionCovariance(topic))
        data["covariance_vx"].append(getVelocityCovariance(topic)[0])
    return data



class TrackingParser:
    """parse tracking object from bagfile
    """
    def __init__(self, bagfile: str, tracking_topic = "/perception/object_recognition/tracking/objects") -> None:
        self.data = getTrackingData(bagfile, tracking_topic)
        self.track_ids = list(self.data.keys())        

    def plot_kinematics(self, x_key, y_key, ax = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        # extract plt_style from kwargs
        plt_style = kwargs.pop("plt_style", "x-")

        for id in self.track_ids:
            dict_data = getTrackerKinematicsDict(self.data[id])
            ax.plot(dict_data[x_key], dict_data[y_key], plt_style)
        return ax
    
    def plot_data(self, x_key = "time", y_keys = [],**kwargs):
        if len(y_keys) == 0:
            y_keys = ["x", "y", "yaw", "vx", "covariance_x", "covariance_vx"]
        
        legend = ["track_" + str(i) for i in range(len(self.track_ids))]

        cols = int(kwargs.pop("cols", 2))
        rows = len(y_keys)//cols + len(y_keys)%cols
        figsize = (8*cols, 5 * rows)
        fig, axs = plt.subplots(rows, cols, sharex=True, figsize=figsize)
        axs = axs.reshape(-1)
        for i, y_key in enumerate(y_keys):
            self.plot_kinematics(x_key, y_key, ax=axs[i], **kwargs)   
            axs[i].set_title(y_key)
            axs[i].set_xlabel("time [s]")
            axs[i].grid()
            axs[i].legend(legend)
        return fig, axs    



def getDetectionData(bagfile, topic_name = "/perception/object_recognition/detection/objects"):
    topic_dict = get_topics_as_dict(bagfile, [topic_name])[topic_name]
    detections = []
    for stamp, msg in topic_dict:
        time = stamp
        for obj in msg.objects:
            detections.append([time, obj])
    return detections

def getDetectionKinematicsDicts(detections: list):
    data = {}
    keys = ["time", "x", "y", "yaw", "vx", "length", "width"]
    for key in keys:
        data[key] = []
    
    for time, topic in detections:
        xy = get2DPosition(topic)
        yaw = getYaw(topic)
        vx = getVX(topic)
        length = topic.shape.dimensions.x
        width = topic.shape.dimensions.y
        data["time"].append(time)
        data["x"].append(xy[0])
        data["y"].append(xy[1])
        data["yaw"].append(yaw)
        data["vx"].append(vx)
        data["length"].append(length)
        data["width"].append(width)
    return data

class DetectionParser:
    def __init__(self, bagfile: str, topic_name = "/perception/object_recognition/detection/objects") -> None:
        detections = getDetectionData(bagfile, topic_name)
        self.data = getDetectionKinematicsDicts(detections)

    def plot_kinematics(self, x_key, y_key, ax = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        plt_style = kwargs.pop("plt_style", "k.")
        plt.plot(self.data[x_key], self.data[y_key], plt_style)
    
    def plot_data(self, x_key = "time", y_keys = [],**kwargs):
        if len(y_keys) == 0:
            y_keys = ["x", "y", "yaw", "vx", "length", "width"]
        
        cols = int(kwargs.pop("cols", 2))
        rows = len(y_keys)//cols + len(y_keys)%cols
        figsize = (8*cols, 5 * rows)
        fig, axs = plt.subplots(rows, cols, sharex=True, figsize=figsize)
        axs = axs.reshape(-1)
        for i, y_key in enumerate(y_keys):
            self.plot_kinematics(x_key, y_key, ax=axs[i], **kwargs)   
            axs[i].set_title(y_key)
            axs[i].set_xlabel("time [s]")
            axs[i].grid()
        return fig, axs
    
class DetctionAndTrackingParser:
    def __init__(self, bagfile:str, topic_names = []) -> None:
        if topic_names == []:
            topic_names = ["/perception/object_recognition/detection/objects", "/perception/object_recognition/tracking/objects"]

        topics_dict, tf_buffer = get_topics_and_tf(bagfile, topic_names)
        self.data = {}
        for topic_name in topic_names:
            self.data[topic_name] = []
        self.max_time = 0
        self.min_time = 1e10
        # parse each data
        for topic_name in topic_names:
            topics = topics_dict[topic_name]
            for stamp, msg in topics:
                time = stamp
                self.max_time = max(self.max_time, time)
                self.min_time = min(self.min_time, time)
                for obj in msg.objects:
                    self.data[topic_name].append([time, obj])
        self.tf_buffer = tf_buffer

    def calc_self_pose(self):
        # get transform
        max_obj_num = 100
        obj_num = min(max_obj_num, (self.max_time - self.min_time)*2)  # assume topic is 2Hz
        obj_num = max(obj_num, 1)
        # virtual timestamp
        times = np.linspace(self.min_time, self.max_time, obj_num)
        self.self_poses = []

        for time in times:
            try:
                pose = get_pose_from_tf(self.tf_buffer, "base_link", "map", time)
                self.self_poses.append([time, pose])
            except:
                continue # do nothing
        if self.self_poses == []:
            raise Exception("No tf found self vehicle pose")
