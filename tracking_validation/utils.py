"""load ros parameters

Only write basic functions to load ros parameters from yaml file.
"""
import numpy as np
from sys import exit

# if ROS1
#from tf.transformations import euler_from_quaternion
# if ROS2
from tf_transformations import euler_from_quaternion

# target objects
from autoware_auto_perception_msgs.msg import TrackedObject, DetectedObject, PredictedObject


def getPosition(object):
    """get pose from ros message
    """
    dtype = type(object)
    if dtype == TrackedObject or dtype == DetectedObject:
        pose = object.kinematics.pose_with_covariance.pose
    elif dtype == PredictedObject:
        pose = object.kinematics.initial_pose_with_covariance.pose
    else:
        print("Invalid object type")
        exit(1)
    return np.array([pose.position.x, pose.position.y, pose.position.z])

def get2DPosition(object):
    """get 2D pose from ros message
    """
    return getPosition(object)[:2]
    
def get2DPositionCovarianceMatrix(object):
    """ get position covariance from ros message
    """
    dtype = type(object)
    if dtype == TrackedObject or dtype == DetectedObject:
        pose_with_covariance = object.kinematics.pose_with_covariance
    elif dtype == PredictedObject:
        pose_with_covariance = object.kinematics.initial_pose_with_covariance
    else:
        print("Invalid object type")
        exit(1)
    return np.array([pose_with_covariance.covariance[0],pose_with_covariance.covariance[1], pose_with_covariance.covariance[6], pose_with_covariance.covariance[7]])

def get2DPositionCovariance(object):
    """get diag x, y, yaw covariance from ros message
    """
    dtype = type(object)
    if dtype == TrackedObject or dtype == DetectedObject:
        pose_with_covariance = object.kinematics.pose_with_covariance
    elif dtype == PredictedObject:
        pose_with_covariance = object.kinematics.initial_pose_with_covariance
    else:
        print("Invalid object type")
        exit(1)
    return np.array([pose_with_covariance.covariance[0],pose_with_covariance.covariance[7], pose_with_covariance.covariance[35]])

def getVelocityCovariance(object):
    dtype = type(object)
    if dtype == TrackedObject or dtype == DetectedObject:
        twist_with_covariance = object.kinematics.twist_with_covariance
    elif dtype == PredictedObject:
        twist_with_covariance = object.kinematics.initial_twist_with_covariance
    else:
        print("Invalid object type")
        exit(1)
    return np.array([twist_with_covariance.covariance[0],twist_with_covariance.covariance[7], twist_with_covariance.covariance[14],
                    twist_with_covariance.covariance[21],twist_with_covariance.covariance[28], twist_with_covariance.covariance[35]])

def getOrientation(object):
    """get orientation from ros message
    """
    dtype = type(object)
    if dtype == TrackedObject or dtype == DetectedObject:
        orientation = object.kinematics.pose_with_covariance.pose.orientation
    elif dtype == PredictedObject:
        orientation = object.kinematics.initial_pose_with_covariance.pose.orientation
    else:
        print("Invalid object type")
        exit(1)
    return euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w], axes='sxyz')

def getYaw(object):
    """get yaw from ros message
    """
    return getOrientation(object)[2]

def getVX(object):
    """get velocity x from ros message
    """
    dtype = type(object)
    if dtype == TrackedObject or dtype == DetectedObject:
        return object.kinematics.twist_with_covariance.twist.linear.x
    elif dtype == PredictedObject:
        return object.kinematics.initial_twist_with_covariance.twist.linear.x
    else:
        print("Invalid object type")
        exit(1)

def getTimeFromStamp(stamp):
    """get time from ros message
    """
    try:
        return stamp.to_nsec()
    except:
        return None
    
def getLabel(obj):
    max_prob = 0
    label = None
    for cl in obj.classification:
        if cl.probability >= max_prob:
            max_prob = cl.probability
            label = cl.label
    return label

def calc2DBboxArea(object):
    """calculate area from detected or tracked object
    """
    if object.shape.type == 0:
        return object.shape.dimensions.x * object.shape.dimensions.y
    else:
        # todo calc polygon with object.shape.footprint 
        return calc2DBboxArea(object.shape.footprint)
    

def calc2DPolygonArea(footprints):
    """calculate area from polygon
    """
    x = footprints[:,0]
    y = footprints[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


# get transform from tf
from rclpy.time import Time
def get_transform_from_tf(tf_buffer, child_frame_id: str, parent_frame_id: str, stamp):
    if type(stamp) == float:
        target_time = Time(second = stamp)
    else:
        target_time = stamp
    try:
        transform = tf_buffer.lookup_transform(child_frame_id, parent_frame_id, target_time)
        return transform
    except:
        return None
    
# get position from transform
from geometry_msgs.msg import PoseWithCovariance
def get_pose_from_tansform(transform):
    pose = PoseWithCovariance()
    pose.pose.position.x = transform.transform.translation.x
    pose.pose.position.y = transform.transform.translation.y
    pose.pose.position.z = transform.transform.translation.z
    pose.pose.orientation.x = transform.transform.rotation.x
    pose.pose.orientation.y = transform.transform.rotation.y
    pose.pose.orientation.z = transform.transform.rotation.z
    pose.pose.orientation.w = transform.transform.rotation.w
    return pose

def get_pose_from_tf(tf_buffer, child_frame_id: str, parent_frame_id: str, stamp):
    transform = get_transform_from_tf(tf_buffer, child_frame_id, parent_frame_id, stamp)
    if transform == None:
        return None
    return get_pose_from_tansform(transform)
