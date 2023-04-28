"""load ros parameters

Only write basic functions to load ros parameters from yaml file.
"""
import numpy as np
from sys import exit

# if ROS1
#from tf.transformations import euler_from_quaternion
# if ROS2
from tf_transformations import euler_from_quaternion

def getPosition(object):
    """get pose from ros message
    """
    pose = object.kinematics.pose_with_covariance.pose
    return np.array([pose.position.x, pose.position.y, pose.position.z])

def get2DPosition(object):
    """get 2D pose from ros message
    """
    return getPosition(object)[:2]
    
def get2DPositionCovariance(object):
    """ get position covariance from ros message
    """
    pose_with_covariance = object.kinematics.pose_with_covariance
    return np.array([pose_with_covariance.covariance[0],pose_with_covariance.covariance[1], pose_with_covariance.covariance[6], pose_with_covariance.covariance[7]])

def getOrientation(object):
    """get orientation from ros message
    """
    orientation = object.kinematics.pose_with_covariance.pose.orientation
    return euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

def getYaw(object):
    """get yaw from ros message
    """
    orientation = object.kinematics.pose_with_covariance.pose.orientation
    return euler_from_quaternion(getOrientation(orientation))[2]

def getTimeFromStamp(stamp):
    """get time from ros message
    """
    try:
        return stamp.to_nsec()
    except:
        return None

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

    
