from rosbag_parser import get_topics_as_dict, get_topics_and_tf
import uuid
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd


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
    topic_dict, tf_buffer = get_topics_and_tf(bagfile, [topic_name])
    trackers = getTrackersInID(topic_dict[topic_name])
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
             "covariance_vx", "covariance_yaw", "class_label"]
    for key in keys:
        data[key] = []

    for time, topic in trackers: # topic: TrackedObject
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
        data["class_label"].append(getLabel(topic))
    return data



class TrackingParser:
    """parse tracking object from bagfile
    """
    def __init__(self, bagfile: str, tracking_topic = "/perception/object_recognition/tracking/objects") -> None:
        self.data = getTrackingData(bagfile, tracking_topic)
        self.track_ids = list(self.data.keys())
        self.df = self.dict_to_dataframe(self.data)

    def dict_to_dataframe(self, data):
        out_df = pd.DataFrame()
        for obj_id in data.keys():
            dict_data = getTrackerKinematicsDict(data[obj_id])
            df = pd.DataFrame(dict_data)
            # fill id columns with obj_id
            df["id"] = obj_id
            # vertically concatenate dataframes
            out_df = pd.concat([out_df, df], axis=0)
        return out_df

            
    def plot_kinematics(self, df, x_key, y_key, ax = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        # extract plt_style from kwargs
        plt_style = kwargs.pop("plt_style", "x-")

        for id in df["id"].unique():
            data = df[df["id"] == id]
            ax.plot(data[x_key], data[y_key], plt_style)
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
            self.plot_kinematics(self.df, x_key, y_key, ax=axs[i], **kwargs)   
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
        """parse detection and tracking msg in rosbag file

        Args:
            bagfile (str): ros2 bag file
            topic_names (list, optional): Topic names to be parsed. Detection/Tracking/Prediction is supported. Defaults is [] and this is converted to "/perception/object_recognition/detection/objects", "/perception/object_recognition/tracking/objects".
        """
        if topic_names == []:
            topic_names = ["/perception/object_recognition/detection/objects", "/perception/object_recognition/tracking/objects"]

        topics_dict, tf_buffer = get_topics_and_tf(bagfile, topic_names)
        self.tf_buffer = tf_buffer
        print("DetectionAndTrackingParser: finish parse rosbag file")

        self.data = {}
        for topic_name in topic_names:
            self.data[topic_name] = []
        self.max_time = 0
        self.min_time = 1e10
        # parse each data
        print("DetectionAndTrackingParser: parsing data")
        for topic_name in tqdm(topic_names):
            topics = topics_dict[topic_name]
            for stamp, msg in topics:
                time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 # use message time
                self.max_time = max(self.max_time, time)
                self.min_time = min(self.min_time, time)

                obj_frame = msg.header.frame_id
                map_frame = "map"
                for obj in msg.objects:
                    # transform to map frame
                    obj = self.transform_object(map_frame, obj_frame, obj, msg.header.stamp)
                    if obj is None:
                        continue
                    self.data[topic_name].append([time, obj])


    def calc_self_pose(self):
        """get self vehicle pose from tf
        """
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
        
    def transform_pose_to_frames(self, target_frame:str, pose_frame:str, pose_with_cov: PoseWithCovariance, stamp):
        """transform pose from pose_frame to target_frame

        Args:
            target_frame (str): target frame for object, usually "map"
            pose_frame (str): pose frame of current object, usually "base_link"
            pose_with_cov (PoseWithCovariance): pose
            stamp (_type_): time stamp of pose

        Returns:
            pose (PoseWithCovariance): pose in target_frame
        """
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from geometry_msgs.msg import TransformStamped

        pose_cov_stamped = PoseWithCovarianceStamped()
        pose_cov_stamped.pose = pose_with_cov
        pose_cov_stamped.header.frame_id = pose_frame
        pose_cov_stamped.header.stamp = stamp

        # get pose from tf from pose_frame to target_frame
        transform = get_transform_from_tf(self.tf_buffer, target_frame, pose_frame, stamp)
        if transform is None:
            #print("transform is None")
            return None
        
        # transform to get pose in target_frame
        # write your code here
        import tf2_geometry_msgs
        # pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, transform)
        transformed_pose_cov_stamped= tf2_geometry_msgs.do_transform_pose_with_covariance_stamped(pose_cov_stamped, transform)
        return transformed_pose_cov_stamped.pose

    def transform_object(self, target_frame:str, obj_frame: str, object, stamp):
        """transform detection/tracking/prediction objects

        Args:
            target_frame (str): output object frame. usually "map"
            obj_frame (str): input object frame. usually "base_link"
            object (_type_): input object. detection/tracking/prediction object in autoware
            stamp (_type_): timestamp in header
        """
        if obj_frame == target_frame:
            return object
        transformed_pose_with_cov = self.transform_pose_to_frames(target_frame, obj_frame, object.kinematics.pose_with_covariance, stamp)

        if transformed_pose_with_cov is None:
            return None
        else:
            object.kinematics.pose_with_covariance = transformed_pose_with_cov
            return object