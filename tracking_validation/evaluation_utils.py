import argparse
import logging
import tempfile
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.manager import PerceptionEvaluationManager
from perception_eval.tool import PerceptionAnalyzer3D
from perception_eval.util.debug import format_class_for_log
from perception_eval.util.debug import get_objects_with_difference
from perception_eval.util.logger_config import configure_logger

# eval object classes
import driving_log_replayer.perception_eval_conversions as eval_conversions
from perception_eval.common.status import FrameID
from perception_eval.common.object import DynamicObject
from autoware_auto_perception_msgs.msg import TrackedObject, TrackedObjects, DetectedObject, DetectedObjects, PredictedObject, PredictedObjects
PerceptionObject = Union[TrackedObject, DetectedObject, PredictedObject]
PerceptionObjects = Union[TrackedObjects, DetectedObjects, PredictedObjects]

# label conversion
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import Label
from autoware_auto_perception_msgs.msg import ObjectClassification
from perception_eval.common.dataset import FrameGroundTruth

# conversion utils
def classification_to_label(classification: ObjectClassification) -> Label:
    """convert perception_msgs ObjectClassification to AutowareLabel

    Args:
        classification (ObjectClassification): _description_

    Returns:
        Label: _description_
    """
    conversion_dict = {
        ObjectClassification.UNKNOWN: Label(AutowareLabel.UNKNOWN,"unknown",[]),
        ObjectClassification.BUS: Label(AutowareLabel.BUS,"bus",[]),
        ObjectClassification.CAR: Label(AutowareLabel.CAR,"car",[]),
        ObjectClassification.TRUCK: Label(AutowareLabel.TRUCK,"truck",[]),
        ObjectClassification.TRAILER: Label(AutowareLabel.TRUCK,"trailer",[]),
        ObjectClassification.MOTORCYCLE: Label(AutowareLabel.MOTORBIKE,"motorbike",[]),
        ObjectClassification.BICYCLE: Label(AutowareLabel.BICYCLE,"bicycle",[]),
        ObjectClassification.PEDESTRIAN: Label(AutowareLabel.PEDESTRIAN,"pedestrian",[]),
    }
    if classification.label not in conversion_dict:
        return None
    return conversion_dict[classification.label]


def get_most_probable_classification(
    array_classification: List[ObjectClassification],
) -> ObjectClassification:
    highest_probability = 0.0
    highest_classification = None
    for classification in array_classification:
        if classification.probability >= highest_probability:
            highest_probability = classification.probability
            highest_classification = classification
    return highest_classification


def list_dynamic_object_from_ros_msg(
    frame_id: FrameID, unix_time: int, objects: Union[List[DetectedObject], List[TrackedObject]]
) -> List[DynamicObject]:
    estimated_objects: List[DynamicObject] = []
    for perception_object in objects:
        most_probable_classification = get_most_probable_classification(
            perception_object.classification
        )
        label = classification_to_label(most_probable_classification)

        uuid = None
        if isinstance(perception_object, TrackedObject):
            uuid = eval_conversions.uuid_from_ros_msg(perception_object.object_id.uuid)

        estimated_object = DynamicObject(
            unix_time=unix_time,
            frame_id=frame_id,
            position=eval_conversions.position_from_ros_msg(
                perception_object.kinematics.pose_with_covariance.pose.position
            ),
            orientation=eval_conversions.orientation_from_ros_msg(
                perception_object.kinematics.pose_with_covariance.pose.orientation
            ),
            size=eval_conversions.dimensions_from_ros_msg(perception_object.shape.dimensions),
            velocity=eval_conversions.velocity_from_ros_msg(
                perception_object.kinematics.twist_with_covariance.twist.linear
            ),
            semantic_score=most_probable_classification.probability,
            semantic_label=label,
            uuid=uuid,
        )
        estimated_objects.append(estimated_object)
    return estimated_objects


# tracked obj to dynamic obj
def tracked_obj_to_dynamic_obj(tracked_obj: TrackedObjects) -> List[DynamicObject]:
    unix_time: int = eval_conversions.unix_time_from_ros_msg(tracked_obj.header)
    frame_id: FrameID = FrameID.from_value(tracked_obj.header.frame_id)
    return list_dynamic_object_from_ros_msg(frame_id, unix_time, tracked_obj.objects)


import numpy as np
# create ground truth frame
def tracked_obj_to_frame_ground_truth(tracked_obj: TrackedObjects, frame_name: str, ego2map: np.ndarray) -> FrameGroundTruth:
    unix_time: int = eval_conversions.unix_time_from_ros_msg(tracked_obj.header)
    dynamic_objects: List[DynamicObject] = tracked_obj_to_dynamic_obj(tracked_obj)
    return FrameGroundTruth(unix_time, frame_name, dynamic_objects, ego2map)

import tf_transformations as tf

def transform_to_matrix(transform_stamped):
    """
    Convert a TransformStamped message to a homogeneous transformation matrix.
    
    Args:
    transform_stamped (geometry_msgs.msg.TransformStamped): the transform message
    
    Returns:
    np.array (shape [4, 4]): the homogeneous transformation matrix
    """
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation

    translation = np.array([t.x, t.y, t.z])
    rotation = tf.quaternion_matrix([q.x, q.y, q.z, q.w])

    matrix = rotation.copy()
    matrix[:3, 3] = translation

    return matrix


# sample実行
class PerceptionLSimMoc:
    def __init__(
        self,
        dataset_paths: List[str],
        evaluation_task: str,
        result_root_directory: str,
    ):
        evaluation_config_dict = {
            "evaluation_task": evaluation_task,
            # ラベル，max x/y，マッチング閾値 (detection/tracking/predictionで共通)
            "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
            "ignore_attributes": ["cycle_state.without_rider"],
            # max x/y position or max/min distanceの指定が必要
            # # max x/y position
            # "max_x_position": 102.4,
            # "max_y_position": 102.4,
            # max/min distance
            "max_distance": 102.4,
            "min_distance": 10.0,
            # # confidenceによるフィルタ (Optional)
            # "confidence_threshold": 0.5,
            # # GTのuuidによるフィルタ (Optional)
            # "target_uuids": ["foo", "bar"],
            # objectごとにparamを設定
            "center_distance_thresholds": [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            # objectごとに同じparamの場合はこのような指定が可能
            "plane_distance_thresholds": [2.0, 3.0],
            "iou_2d_thresholds": [0.5],
            "iou_3d_thresholds": [0.5],
            "min_point_numbers": [0, 0, 0, 0],
        }

        evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            dataset_paths=dataset_paths,
            frame_id="base_link" if evaluation_task == "detection" else "map",
            merge_similar_labels=False,
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
            load_raw_data=True,
        )

        _ = configure_logger(
            log_file_directory=evaluation_config.log_directory,
            console_log_level=logging.INFO,
            file_log_level=logging.INFO,
        )

        self.evaluator = PerceptionEvaluationManager(evaluation_config=evaluation_config)

    def callback(
        self,
        unix_time: int,
        estimated_objects: List[DynamicObject],
    ) -> None:

        # 現frameに対応するGround truthを取得
        ground_truth_now_frame = self.evaluator.get_ground_truth_now_frame(unix_time)

        # [Option] ROS側でやる（Map情報・Planning結果を用いる）UC評価objectを選別
        # ros_critical_ground_truth_objects : List[DynamicObject] = custom_critical_object_filter(
        #   ground_truth_now_frame.objects
        # )
        ros_critical_ground_truth_objects = ground_truth_now_frame.objects

        # 1 frameの評価
        # 距離などでUC評価objectを選別するためのインターフェイス（PerceptionEvaluationManager初期化時にConfigを設定せず、関数受け渡しにすることで動的に変更可能なInterface）
        # どれを注目物体とするかのparam
        critical_object_filter_config: CriticalObjectFilterConfig = CriticalObjectFilterConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            ignore_attributes=["cycle_state.without_rider"],
            max_x_position_list=[30.0, 30.0, 30.0, 30.0],
            max_y_position_list=[30.0, 30.0, 30.0, 30.0],
        )
        # Pass fail を決めるパラメータ
        frame_pass_fail_config: PerceptionPassFailConfig = PerceptionPassFailConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            matching_threshold_list=[2.0, 2.0, 2.0, 2.0],
        )

        frame_result = self.evaluator.add_frame_result(
            unix_time=unix_time,
            ground_truth_now_frame=ground_truth_now_frame,
            estimated_objects=estimated_objects,
            ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
        )
        self.visualize(frame_result)

    def get_final_result(self) -> MetricsScore:
        """
        処理の最後に評価結果を出す
        """

        # use case fail object num
        number_use_case_fail_object: int = 0
        for frame_results in self.evaluator.frame_results:
            number_use_case_fail_object += frame_results.pass_fail_result.get_fail_object_num()
        logging.info(f"final use case fail object: {number_use_case_fail_object}")
        final_metric_score = self.evaluator.get_scene_result()

        # final result
        logging.info(f"final metrics result {final_metric_score}")
        return final_metric_score

    def visualize(self, frame_result: PerceptionFrameResult):
        """
        Frameごとの可視化
        """
        logging.info(
            f"{len(frame_result.pass_fail_result.tp_object_results)} TP objects, "
            f"{len(frame_result.pass_fail_result.fp_object_results)} FP objects, "
            f"{len(frame_result.pass_fail_result.fn_objects)} FN objects",
        )

        if frame_result.metrics_score.maps[0].map < 0.7:
            logging.debug("mAP is low")
            # logging.debug(f"frame result {format_class_for_log(frame_result.metrics_score)}")

        # Visualize the latest frame result
        # self.evaluator.visualize_frame()

if __name__ == "__main__":

    # tracking eval
    result_root_directory: str = "temp/tracking_eval/"
    tracking_lsim = PerceptionLSimMoc([], "tracking", result_root_directory)