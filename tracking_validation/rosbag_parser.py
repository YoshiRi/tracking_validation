#!/usr/bin/env /usr/bin/python3
"""
Author: yoshi.ri@tier4.jp

Reference
- https://github.com/ros2/rosbag2/blob/c7c7954d4d9944c160d7b3d716d1cb95d34e37e4/rosbag2_py/test/test_sequential_writer.py
- https://github.com/tier4/ros2bag_extensions/blob/main/ros2bag_extensions/ros2bag_extensions/verb/filter.py
"""
import ros2bag
import rclpy
import uuid

from datetime import datetime
from rosbag2_py import *

from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import tf2_ros


def create_reader(bag_dir: str) -> SequentialReader:
    storage_options = get_default_storage_options(bag_dir)
    converter_options = get_default_converter_options()

    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    return reader

def get_default_converter_options() -> ConverterOptions:
    return ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )


def get_default_storage_options(uri: str) -> StorageOptions:
    return StorageOptions(
        uri=uri,
        storage_id="sqlite3",
    )


def get_starting_time(uri: str) -> datetime:
    info = Info().read_metadata(uri, "sqlite3")
    return info.starting_time


def get_topics_as_dict(bagfile, topic_list: list):
    reader = create_reader(bagfile)
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}


    topics = {}
    for topic in topic_list:
        topics[topic] = []

    while reader.has_next():
        topic_name, data, stamp = reader.read_next()
        if topic_name in topic_list:
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)
            topics[topic_name].append([stamp, msg])

    return topics


# tf buffer

def get_tf_listener(bagfile:str):

    tf_buffer = tf2_ros.Buffer()
    #tf_listener = tf2_ros.TransformListener(tf_buffer)

    reader = create_reader(bagfile)
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    while reader.has_next():
        topic_name, data, stamp = reader.read_next()
        if topic_name == "/tf":
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)
            for transform in msg.transforms:
                tf_buffer.set_transform(transform, "default_authority")

    return tf_buffer

def get_topics_and_tf(bagfile:str, topic_list: list):
    tf_buffer = tf2_ros.Buffer()

    reader = create_reader(bagfile)
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}


    topics = {}
    for topic in topic_list:
        topics[topic] = []

    while reader.has_next():
        topic_name, data, stamp = reader.read_next()
        if topic_name in topic_list:
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)
            topics[topic_name].append([stamp, msg])
        elif topic_name == "/tf":
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)
            for transform in msg.transforms:
                tf_buffer.set_transform(transform, "default_authority")

    return topics, tf_buffer