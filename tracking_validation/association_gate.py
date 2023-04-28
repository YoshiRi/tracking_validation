import math
import numpy as np
from scipy.spatial.distance import mahalanobis
from utils import *
from iou_utils import get_2d_iou

from autoware_auto_perception_msgs.msg import DetectedObject
from autoware_auto_perception_msgs.msg import DetectedObjectKinematics
from autoware_auto_perception_msgs.msg import ObjectClassification
from autoware_auto_perception_msgs.msg import Shape

def association_gate(tracker, detection, max_dist, min_area, max_area, max_angle):
    """
    Checks if the tracker and the detected object pass the association gates.

    Args:
        tracker (Tracker): tracker object of TrackedObject
        detection (Detection): detection object of DetectedObject
        max_dist (float): maximum distance threshold
        min_area (float): minimum area threshold
        max_area (float): maximum area threshold
        max_angle (float): maximum angle threshold in radians

    Returns:
        dict: True/False for each gate
    """

    tracker_pos = get2DPosition(tracker)
    tracker_covariance = getPositionCovariance(tracker)
    detection_pos = get2DPosition(detection)
    detection_area = calc2DBboxArea(detection)
    tracker_orientation = getYaw(tracker) 
    detection_orientation = getYaw(detection)
    tracker_bbox = (tracker.x_min, tracker.y_min, tracker.x_max, tracker.y_max)
    detection_bbox = (detection.x_min, detection.y_min, detection.x_max, detection.y_max)
    min_iou = 0.01

    return {
            "distance_gate": distance_gate(tracker_pos, detection_pos, max_dist),
            "area_gate": area_gate(detection_area, min_area, max_area),
            "angle_gate": angle_gate(tracker_orientation, detection_orientation, max_angle),
            "mahalanobis_distance_gate": mahalanobis_distance_gate(tracker_pos, detection_pos, tracker_covariance, 3),
            "iou_gate": iou_gate(tracker_bbox, detection_bbox, min_iou)
            }


def distance_gate(tracker_pos, detection_pos, max_dist):
    """
    Checks if the distance between the tracker and the detected object is within a maximum distance threshold.

    Args:
        tracker_pos (tuple): x, y position of the tracker
        detection_pos (tuple): x, y position of the detected object
        max_dist (float): maximum distance threshold

    Returns:
        bool: True if the distance is within the threshold, False otherwise
    """
    dist = math.sqrt((tracker_pos[0] - detection_pos[0])**2 + (tracker_pos[1] - detection_pos[1])**2)
    return dist <= max_dist


def area_gate(detection_area, min_area, max_area):
    """
    Checks if the area of the detected object is within a minimum and maximum area threshold.

    Args:
        detection_area (float): area of the detected object
        min_area (float): minimum area threshold
        max_area (float): maximum area threshold

    Returns:
        bool: True if the area is within the thresholds, False otherwise
    """
    return min_area <= detection_area <= max_area

def angle_gate(tracker_orientation, detection_orientation, max_angle):
    """
    Checks if the angle between the orientation of the tracker and the detected object is within a maximum angle threshold.

    Args:
        tracker_orientation (float): orientation of the tracker in radians
        detection_orientation (float): orientation of the detected object in radians
        max_angle (float): maximum angle threshold in radians

    Returns:
        bool: True if the angle is within the threshold, False otherwise
    """
    angle_diff = abs(tracker_orientation - detection_orientation)
    angle_diff = min(angle_diff, 2*math.pi - angle_diff)
    return angle_diff <= max_angle

def mahalanobis_distance_gate(tracker_pos, detection_pos, tracker_covariance, threshold):
    """
    Checks if the Mahalanobis distance between the tracker and the detected object is within a threshold.

    Args:
        tracker_pos (tuple): x, y position of the tracker
        detection_pos (tuple): x, y position of the detected object
        tracker_covariance (np.ndarray): 2x2 covariance matrix of the tracker position
        threshold (float): maximum Mahalanobis distance threshold

    Returns:
        bool: True if the distance is within the threshold, False otherwise
    """
    mahalanobis_dist = mahalanobis(tracker_pos, detection_pos, np.linalg.inv(tracker_covariance))
    return mahalanobis_dist <= threshold

def iou_gate(tracker, detection, min_iou=0.01):
    """
    Checks if the intersection over union (IoU) between the tracker and the detected object is above a minimum threshold.

    Args:
        tracker: tracker
        detection: detected object
        min_iou (float): minimum IoU threshold

    Returns:
        bool: True if the IoU is above the threshold, False otherwise
    """
    iou = get_2d_iou(tracker, detection)
    return iou >= min_iou