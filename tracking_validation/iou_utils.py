import math
import numpy as np
from autoware_auto_perception_msgs.msg import Shape, DetectedObject
import geometry_msgs.msg as gm
from shapely.geometry import Polygon
from tf_transformations import euler_from_quaternion

def OrientationToYaw(orientation):
    """_summary_

    Args:
        orientation (gm.Quaternion): _description_

    Returns:
        float: yaw
    """
    return euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])[2]

def calc_offset_point(pose: gm.Pose, x: float, y: float):
    """_summary_

    Args:
        pose (gm.Pose): _description_
        x (float): _description_
        y (float): _description_
        z (float): _description_

    Returns:
        [x, y]: point
    """
    yaw = OrientationToYaw(pose.orientation)

    x_ = pose.position.x + np.cos(yaw) * x - np.sin(yaw) * y
    y_ = pose.position.y + np.sin(yaw) * x + np.cos(yaw) * y
    return np.array([x_, y_])


def to_polygon_2d(pose: gm.Pose, shape: Shape) -> Polygon:
    polygon = []

    if shape.type == Shape.BOUNDING_BOX:
        point0 = calc_offset_point(pose, shape.dimensions.x / 2.0, shape.dimensions.y / 2.0)
        point1 = calc_offset_point(pose, -shape.dimensions.x / 2.0, shape.dimensions.y / 2.0)
        point2 = calc_offset_point(pose, -shape.dimensions.x / 2.0, -shape.dimensions.y / 2.0)
        point3 = calc_offset_point(pose, shape.dimensions.x / 2.0, -shape.dimensions.y / 2.0)
        polygon = [point0, point1, point2, point3]
        polygon = Polygon(np.array(polygon).reshape(4, 2))

    elif shape.type == Shape.CYLINDER:
        radius = shape.dimensions.x / 2.0
        circle_discrete_num = 6
        for i in range(circle_discrete_num):
            point = np.array(2)
            point[0] = math.cos((i / circle_discrete_num) * 2.0 * math.pi + math.pi / circle_discrete_num) * radius + pose.position.x
            point[1] = math.sin((i / circle_discrete_num) * 2.0 * math.pi + math.pi / circle_discrete_num) * radius + pose.position.y
            polygon.append(point)
        polygon = Polygon(np.array(polygon).reshape(-1,2))

    elif shape.type == Shape.POLYGON:
        
        poly_yaw = OrientationToYaw(pose.orientation)
        polygon = Polygon(shape.footprint.points)
    else:
        raise ValueError("The shape type is not supported.")

    return polygon


def get_sum_area(polygons) -> float:
    return sum([p.area for p in polygons])

def get_convex_shape_area(source_polygon, target_polygon) -> float:
    union_polygon = Polygon.union(source_polygon, target_polygon)
    convex_polygon = union_polygon.convex_hull
    return convex_polygon.area

def get_intersection_area(source_polygon: Polygon, target_polygon: Polygon) -> float:
    """get polygon intersection

    Args:
        source_polygon (_type_): np.array([[x1,y1],[x2,y2],...])
        target_polygon (_type_): np.array([[x1,y1],[x2,y2],...])
    """
    intersection_polygon = Polygon.intersection(source_polygon, target_polygon)
    return intersection_polygon.area


def get_union_area(source_polygon: Polygon, target_polygon: Polygon) -> float:
    """get polygon union

    Args:
        source_polygon (_type_): np.array([[x1,y1],[x2,y2],...])
        target_polygon (_type_): np.array([[x1,y1],[x2,y2],...])
    """
    union_polygon = Polygon.union(source_polygon, target_polygon)
    return union_polygon.area


def get_2d_iou(source_object:DetectedObject, target_object:DetectedObject, min_union_area=0.01):
    source_polygon = to_polygon_2d(source_object.kinematics.pose_with_covariance.pose, source_object.shape)
    target_polygon = to_polygon_2d(target_object.kinematics.pose_with_covariance.pose, target_object.shape)
    intersection_area = get_intersection_area(source_polygon, target_polygon)
    if intersection_area == 0.0:
        return 0.0
    union_area = get_union_area(source_polygon, target_polygon)
    if union_area < min_union_area:
        return 0.0
    iou = min(1.0, intersection_area / union_area)
    return iou

def get_2d_generalized_iou(source_object, target_object):
    source_polygon = to_polygon_2d(source_object.pose, source_object.shape)
    target_polygon = to_polygon_2d(target_object.pose, target_object.shape)
    intersection_area = get_intersection_area(source_polygon, target_polygon)
    union_area = get_union_area(source_polygon, target_polygon)
    convex_shape_area = get_convex_shape_area(source_polygon, target_polygon)
    
    if union_area < 0.01:
        return 0.0
    
    iou = min(1.0, intersection_area / union_area)
    return iou - (convex_shape_area - union_area) / convex_shape_area
