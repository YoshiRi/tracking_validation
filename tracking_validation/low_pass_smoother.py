import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import math
import random
import pandas as pd
import numpy as np
from scipy.fft import fft

# data parser
from tracking_parser import DetctionAndTrackingParser
from utils import *
from autoware_auto_perception_msgs.msg import ObjectClassification 

from typing import List, Dict, Tuple, Union, Optional, Any
from dash_visualizer import DataForVisualize

from matplotlib import pyplot as plt
from copy import deepcopy

from scipy.signal import butter, filtfilt
from tqdm import tqdm


def load_rosbag_data(rosbag_file_path:str, topics:List[str]):
    # parser is written in tracking_parser
    parser = DetctionAndTrackingParser(rosbag_file_path, topics) # parser.data has dict of topic name and data
    # parser.data is a dict of topic name and data
    temp_list = []
    for topic_name in parser.data:
        for topic in  parser.data[topic_name]:
            data_dict = {}
            time = topic[0]
            obj = topic[1]
            viz_data = DataForVisualize()
            viz_data.fromPerceptionObjectsWithTime(obj, time)
            viz_data.setTopicName(topic_name)
            temp_list.append(viz_data.data)
    df = pd.DataFrame(temp_list)
    return df

import numpy as np

def low_pass_df(df:pd.DataFrame, columns:List[str], cutoff_frequency:float = 1.0, order:int=4):
    """
    Apply a zero-phase lowpass filter to specified columns of a Pandas DataFrame.

    :param df: Pandas DataFrame containing the data.
    :param columns: List of columns to which the filter should be applied.
    :param cutoff_frequency: Cutoff frequency of the lowpass filter. [Hz]
    :param sampling_rate: Sampling rate of the data.
    :param order: Order of the Butterworth filter. Default is 3.
    :return: DataFrame with the filtered data.
    """
    # Create a copy of the dataframe to avoid modifying the original data
    filtered_df = df.copy()
    sampling_rate = 1.0 / (df["time"].iloc[1] - df["time"].iloc[0]) # Hz

    # Design the Butterworth lowpass filter
    min_pad_len = 4 * max(order, 1)
    df_len = len(df)
    if df_len < min_pad_len and df_len > 4:
        order = 1
    elif df_len < 4:
        return filtered_df

    b, a = butter(order, cutoff_frequency, btype='lowpass',fs = sampling_rate)

    # Apply the filter to each specified column
    for column in columns:
        filtered_df[column] = filtfilt(b, a, df[column])

    return filtered_df


def moving_window_fft(data:np.ndarray, window_size:int, step_size:int, sampling_rate:float):
    """
    Apply FFT using a moving window approach and return the frequency and amplitude for each window.

    :param data: Input data (time series), as a Pandas Series or NumPy array.
    :param window_size: Size of the moving window (in samples).
    :param step_size: Step size for the moving window (in samples).
    :param sampling_rate: Sampling rate of the data.
    :return: Tuple of (time_points, frequencies, amplitudes), where time_points is the center of each window.
    """
    data = np.asarray(data)  # Ensure data is a NumPy array
    n = len(data)
    frequencies = np.fft.fftfreq(window_size, d=1/sampling_rate)
    time_points = []
    amplitudes = []

    for start in range(0, n - window_size, step_size):
        end = start + window_size
        window_center = (start + end) / 2 / sampling_rate
        time_points.append(window_center)

        windowed_data = data[start:end]
        windowed_data -= np.mean(windowed_data)  # Remove mean to reduce spectral leakage
        fft_result = fft(windowed_data)
        amplitude = np.abs(fft_result)
        amplitudes.append(amplitude)

    return time_points, frequencies, np.array(amplitudes)

def get_interpolated_df(df:pd.DataFrame, columns:List[str], sampling_rate:float = 100.0):
    min_time = df['time'].min()
    max_time = df['time'].max()
    equal_time = np.linspace(min_time, max_time, len(df))
    
    # Interpolating each column of interest
    interpolated_data = {}
    for column in columns:
        interpolated_data[column] = np.interp(equal_time, df['time'], df[column])
    interpolated_data['time'] = equal_time
    return pd.DataFrame(interpolated_data)

def full_fft(data:np.ndarray, sampling_rate:float):
    """
    Apply FFT using a moving window approach and return the frequency and amplitude for each window.

    :param data: Input data (time series), as a Pandas Series or NumPy array.
    :param window_size: Size of the moving window (in samples).
    :param step_size: Step size for the moving window (in samples).
    :param sampling_rate: Sampling rate of the data.
    :return: Tuple of (time_points, frequencies, amplitudes), where time_points is the center of each window.
    """
    data = np.asarray(data)  # Ensure data is a NumPy array
    n = len(data)
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_result = fft(data)
    amplitude = np.abs(fft_result)
    return frequencies, amplitude

def fft_df(df:pd.DataFrame, columns:List[str], window_size:int = 40, step_size:int = 1):
    """
    Apply FFT using a moving window approach to specified columns of a Pandas DataFrame.

    :param df: Pandas DataFrame containing the data.
    :param columns: List of columns to which the filter should be applied.
    :param window_size: Size of the moving window (in samples).
    :param step_size: Step size for the moving window (in samples).
    :param sampling_rate: Sampling rate of the data.
    :return: DataFrame with the filtered data.
    """
    # Create a copy of the dataframe to avoid modifying the original data
    data = {}
    sampling_rate:float = 1.0 / (df["time"].iloc[1] - df["time"].iloc[0]) # Hz

    df_len = len(df)
    if df_len < window_size:
        return data

    # Apply the filter to each specified column
    for column in columns:
        time_points, frequencies, amplitudes = moving_window_fft(df[column].values, window_size, step_size, sampling_rate)
        data[column] = amplitudes
        amplitudes_db = 20. * np.log10(np.array(amplitudes) + 1e-6)
        data[column + "_db"] = amplitudes_db
        # full fft
        frequencies_, amplitudes = full_fft(df[column].values, sampling_rate)
        amplitudes_db = 20. * np.log10(np.array(amplitudes) + 1e-6)
        # only need half of the data
        data[column + "_full_fft"] = amplitudes_db[:len(amplitudes_db)//2]

    data["time"] = np.array(time_points)
    data["original_time"] = df["time"].values
    data["frequencies"] = frequencies
    data["full_frequencies"] = frequencies_[:len(frequencies_)//2]
    return data

def show_moving_fft(data: dict, column:str, labels:List[str]):
    # plt.figure()
    if data == {}:
        return
    try:
        target = column 
        mean_amplitude = np.mean(data[target].flatten())
        std_amplitude = np.std(data[target].flatten())
        plt.pcolormesh(data["time"], data["frequencies"], data[target].T, shading='gouraud',
                    vmin=mean_amplitude - 3. * std_amplitude, vmax=mean_amplitude + 3. * std_amplitude)
        plt.colorbar(label='Amplitude in dB')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title('Moving Window FFT Analysis')
        plt.ylim(0, 5)  # Limiting frequency for better visualization
    except:
        print("error")
        return
    
def show_full_fft(data: dict, column:str, labels:List[str]):
    # plt.figure()
    if data == {}:
        return
    try:
        target = column + "_full_fft"
        plt.plot(data["full_frequencies"][:-1], data[target][:-1], label=labels[0])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title('Full FFT in ' + column)
        plt.xlim(0, 5)  # Limiting frequency for better visualization
    except:
        print("error")
        return


def show_time_series(dfs:List[pd.DataFrame], column:str, labels:List[str]):
    # plt.figure()
    styles = [".","-", "--", "-.", ":"]
    for i in range(len(dfs)):
        df = dfs[i]
        plt.plot(df["time"], df[column], styles[i], label=labels[i])
    plt.xlabel("time [s]")
    plt.ylabel(column)
    plt.title(f"time series of {column}")
    plt.legend()
    plt.grid()

def main(rosbag_file_path: str = "", topics: list = []):
    """main function to run visualizer

    Args:
        rosbag_file_path (str, optional): _description_. Defaults to "".
    """
            # create dummy data
    if rosbag_file_path == "":
        rosbag_file_path = "dummy"
        rosbag_file_path = "/home/yoshiri/autoware_bag/prediction_data/cutindata/202211_cutin_data/no18_ego30k_tag20k_20m_vy0.5ms_b64370f4-8b54-4cdd-863d-11d942f6bc7f_2022-11-10-11-42-57/b64370f4-8b54-4cdd-863d-11d942f6bc7f_2022-11-10-11-42-57_0.db3"
    if topics == []:
        topics = ["/perception/object_recognition/tracking/objects"]

    df_file = "car_df.csv"

    import os
    if os.path.exists(df_file):
        car_df = pd.read_csv(df_file)
    else:
        df = load_rosbag_data(rosbag_file_path, topics)
        print(df["classification"].unique())
        car_labels = [ObjectClassification.CAR, ObjectClassification.TRUCK, ObjectClassification.BUS, ObjectClassification.TRAILER]
        car_df = df[df["classification"].isin(car_labels)]
        # save car_df
        car_df.to_csv("car_df.csv")
    unique_ids = car_df["uuid"].unique()
    # interactive_plot(car_df, unique_ids)
    for id in tqdm(unique_ids):
        selected_df = car_df[car_df["uuid"] == id]
        lpf_df1 = low_pass_df(selected_df, ["x", "yaw", "vx", "omega"], cutoff_frequency=1.0)
        lpf_df2 = low_pass_df(selected_df, ["x", "yaw", "vx", "omega"], cutoff_frequency=0.5)
        plt.figure()
        plt.subplot(4,1,1)
        show_time_series([selected_df, lpf_df1, lpf_df2], "x", ["raw", "lpf 1Hz", "lpf 0.5Hz"])
        plt.subplot(4,1,2)
        show_time_series([selected_df, lpf_df1, lpf_df2], "yaw", ["raw", "lpf 1Hz", "lpf 0.5Hz"])
        plt.subplot(4,1,3)
        show_time_series([selected_df, lpf_df1, lpf_df2], "vx", ["raw", "lpf 1Hz", "lpf 0.5Hz"])
        plt.subplot(4,1,4)
        show_time_series([selected_df, lpf_df1, lpf_df2], "omega", ["raw", "lpf 1Hz", "lpf 0.5Hz"])
        fft_data = fft_df(selected_df, ["x", "yaw", "vx", "omega"])
        plt.figure()
        plt.subplot(2,1,1)
        show_full_fft(fft_data, "x", ["raw", "lpf 1Hz", "lpf 0.5Hz"])
        plt.subplot(2,1,2)
        show_full_fft(fft_data, "yaw", ["raw", "lpf 1Hz", "lpf 0.5Hz"])
    plt.show()


if __name__ == "__main__":
    main()
