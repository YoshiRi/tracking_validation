# apply kalman smoother to the tracking results
# plotly visualizer
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import math
import random
import pandas as pd
import numpy as np

# data parser
from tracking_parser import DetctionAndTrackingParser
from utils import *
from autoware_auto_perception_msgs.msg import ObjectClassification 

from typing import List, Dict, Tuple, Union, Optional, Any
from dash_visualizer import DataForVisualize

from matplotlib import pyplot as plt
from copy import deepcopy


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

class CTRAModel:
    def __init__(self):
        """
        CTRA モデルの初期化
        :param dt: 時間ステップ (秒)

        state model: [x, y, theta, v, omega, a]
        """
        self.state = np.array([0., 0., 0., 0., 0., 0.]).reshape(-1, 1) # state vector
        self.last_update:float = 0
        self.init_thresholds()
        self.Q = np.diag([0.01, 0.01, 0.05, 0.01, 0.05, 0.01])
        self.Rvec = np.array([1.0, 1.0, 0.6, 0.1]).reshape(-1,1) # measurement noise

    def init_state(self,init_time:float, init_state:np.ndarray):
        self.state = init_state
        self.last_update = init_time
        self.P = np.diag([1., 1., 1., 1e2, 1e2, 1e2])

    def init_thresholds(self):
        self.omega_threshold = 1e-1  # used to judge if the object is straight or curving

    def get_jacobian_F(self, state: np.ndarray , dt:float)-> np.ndarray:
        """
        Jacobian of state transition function
        """
        F = np.identity(6)
        x, y, theta, v, omega, a = state.squeeze().tolist()

        if abs(omega) < self.omega_threshold:  # straight
            F[0, 2] = -v * np.sin(theta) * dt
            F[0, 3] = np.cos(theta) * dt
            F[1, 2] = v * np.cos(theta) * dt
            F[1, 3] = np.sin(theta) * dt
            F[2, 4] = dt
            F[3, 5] = dt
        else:  # Curving
            F[0, 2] = (v / omega) * (np.cos(theta + omega * dt) - np.cos(theta))
            F[0, 3] = (1. / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            F[1, 2] = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            F[1, 3] = -(1. / omega) * (np.cos(theta + omega * dt) - np.cos(theta))
            F[2, 4] = dt
            F[3, 5] = dt
        
        return F

    def state_transition(self, state: np.ndarray, dt: float):
        """
        状態遷移関数
        :param state: 現在の状態ベクトル [x, y, theta, v, omega, a]
        :param control: 制御ベクトル [omega_dot, a_dot]
        :return: 次の状態ベクトル
        """
        x, y, theta, v, omega, a = state.squeeze().tolist()

        # 加速度・角速度は減衰させたいので、減衰率を計算
        # 新しい位置と速度を計算
        if abs(omega) < self.omega_threshold:  # straight
            dx = v * np.cos(theta) * dt
            dy = v * np.sin(theta) * dt
            dtheta = omega * dt
            dv = a * dt
        else:  # Curving
            dx = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            dy = (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
            dtheta = omega * dt
            dv = a * dt

        # 状態を更新
        x += dx
        y += dy
        theta += dtheta
        v += dv

        return np.array([x, y, theta, v, omega, a]).reshape(-1, 1)

    def normalize_angle(self, state):
        # restrict angle to [-pi, pi]
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        return state

    def predict(self, time:float):
        dt:float = time - self.last_update
        F = self.get_jacobian_F(self.state, dt)
        self.state = self.state_transition(self.state, dt)
        self.state = self.normalize_angle(self.state)
        self.P = F @ self.P @ F.T + self.Q
        self.last_update = time

    def update(self, z: np.ndarray, R: np.ndarray=None):
        """
        観測更新
        :param z: 観測ベクトル [x, y, theta] or [x, y, theta, v]
        :param R: 観測ノイズの共分散行列
        """
        if R is None:
            R = np.diag(self.Rvec[0:len(z)])
            
        H = np.eye(len(z),6)
        err = z - H @ self.state
        err = self.normalize_angle(err)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ err
        self.state = self.normalize_angle(self.state)
        self.P = (np.identity(6) - K @ H) @ self.P

    def predict_and_update(self, time:float, z: np.ndarray=None, R: np.ndarray=None):
        self.predict(time)
        if z is not None:
            self.update(z, R)
        return self.state, self.P
    

class CTRVModel:
    def __init__(self):
        """
        CTRV モデルの初期化
        :param dt: 時間ステップ (秒)

        state model: [x, y, theta, v, omega]
        """
        self.state = np.array([0., 0., 0., 0., 0.]).reshape(-1, 1)
        self.last_update:float = 0
        self.init_thresholds()
        self.Q = np.diag([0.02, 0.02, 0.05, 0.1, 0.05])
        self.Rvec = np.array([2.0, 2., 0.2, 0.1]).reshape(-1,1) # measurement noise

    def init_state(self,init_time:float, init_state:np.ndarray):
        self.state = init_state
        self.last_update = init_time
        self.P = np.diag([1., 1., 1., 1e2, 1e2])

    def init_thresholds(self):
        self.omega_threshold = 1e5

    def get_jacobian_F(self, state: np.ndarray , dt:float)-> np.ndarray:
        """
        Jacobian of state transition function
        """
        F = np.identity(5)
        x, y, theta, v, omega = state.squeeze().tolist()
    
        F[0, 2] = -v * np.sin(theta) * dt
        F[0, 3] = np.cos(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt
        F[2, 4] = dt
        return F
    
    def state_transition(self, state: np.ndarray, dt: float):
        """
        状態遷移関数
        :param state: 現在の状態ベクトル [x, y, theta, v, omega]
        :param control: 制御ベクトル [omega_dot, a_dot]
        :return: 次の状態ベクトル
        """
        x, y, theta, v, omega = state.squeeze().tolist()

        # 加速度・角速度は減衰させたいので、減衰率を計算
        # 新しい位置と速度を計算
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = omega * dt
        dv = 0

        # 状態を更新
        x += dx
        y += dy
        theta += dtheta
        v += dv

        return np.array([x, y, theta, v, omega]).reshape(-1, 1)
    
    def predict(self, time:float):
        dt:float = time - self.last_update
        F = self.get_jacobian_F(self.state, dt)
        self.state = self.state_transition(self.state, dt)
        self.state = self.normalize_angle(self.state)
        self.P = F @ self.P @ F.T + self.Q
        self.last_update = time

    def normalize_angle(self, state):
        # restrict angle to [-pi, pi]
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        return state

    def update(self, z: np.ndarray, R: np.ndarray=None):
        """
        観測更新
        :param z: 観測ベクトル [x, y, theta] or [x, y, theta, v]
        :param R: 観測ノイズの共分散行列
        """
        if R is None:
            R = np.diag(self.Rvec[0:len(z)])
            
        H = np.eye(len(z),5)
        err = z - H @ self.state
        err = self.normalize_angle(err)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ err
        self.state = self.normalize_angle(self.state)
        self.P = (np.identity(5) - K @ H) @ self.P

    def predict_and_update(self, time:float, z: np.ndarray=None, R: np.ndarray=None):
        self.predict(time)
        if z is not None:
            self.update(z, R)
        return self.state, self.P

class UKFCTRVModel(CTRVModel):
    def __init__(self):
        super().__init__()
        self.alpha = 1e-2
        self.beta = 2
        self.kappa = -3 # 3 - n

    def init_state(self,init_time:float, init_state:np.ndarray):
        self.state = init_state
        self.state = self.normalize_angle(self.state)
        self.last_update = init_time
        self.P = np.diag([100., 100., 100., 10., 10.])

    def generate_sigma_points(self, state:np.ndarray, P:np.ndarray):
        n = len(state)
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = np.linalg.cholesky((n + lambda_) * P)
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = state.squeeze()
        for k in range(n):
            sigma_points[k+1] = state.squeeze() + U[:,k]
            sigma_points[n+k+1] = state.squeeze() - U[:,k]
        return sigma_points # 2n+1 x n
    
    def calc_weights(self, n:int):
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        Wm = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
        Wm[0] = lambda_ / (n + lambda_)
        Wc = np.copy(Wm)
        Wc[0] = Wc[0] + (1 - self.alpha**2 + self.beta)
        return Wm, Wc

    def calc_weighted_mean(self, sigma_points:np.ndarray, Wm:np.ndarray):
        Wm = Wm.reshape(-1,1)
        output = np.sum(Wm * sigma_points, axis=0)
        # recalculate angle summation with atan2
        output[2] = np.arctan2(np.sum(Wm * np.sin(sigma_points[:,2])), np.sum(Wm * np.cos(sigma_points[:,2])))
        return output
    
    def predict(self, time:float):
        dt:float = time - self.last_update
        n = len(self.state)
        sigma_points = self.generate_sigma_points(self.state, self.P)
        Wm, Wc = self.calc_weights(n)
        sigma_points_ = np.array([self.state_transition(sigma_points[i].reshape(-1,1), dt).squeeze() for i in range(2 * n + 1)])
        self.state = self.calc_weighted_mean(sigma_points_, Wm).reshape(-1,1)
        self.state = self.normalize_angle(self.state)
        self.P += np.eye(n) * 1e-9 # add small value to avoid singular matrix
        for i in range(2 * n + 1):
            err = sigma_points_[i].reshape(-1,1) - self.state
            err = self.normalize_angle(err)
            self.P += Wc[i] * np.outer(err, err)
        self.P += self.Q
        self.last_update = time
        return sigma_points_
    
    def measurement_function(self, state:np.ndarray):
        return state[0:3]

    def update(self, sigma_points_:np.ndarray, measurements:np.ndarray, R:np.ndarray=None):
        if R is None:
            R = np.diag(self.Rvec[0:len(measurements)])


        n = len(self.state)
        Wm, Wc = self.calc_weights(n)
        z_sigma_points = np.array([self.measurement_function(sigma_points_[i]) for i in range(2 * n + 1)]) # 2n+1 x 3
        z_pred = self.calc_weighted_mean(z_sigma_points, Wm) # 3 x 1
        z_pred = z_pred[0:len(measurements)].reshape(-1,1)
        Pzz = np.zeros((len(measurements), len(measurements)))
        Pxz = np.zeros((n, len(measurements)))
        for i in range(2 * n + 1):
            err = sigma_points_[i].reshape(-1,1) - self.state
            err = self.normalize_angle(err)
            err_z = self.measurement_function(sigma_points_[i]).reshape(-1,1) - z_pred
            err_z = self.normalize_angle(err_z)
            Pzz += Wc[i] * np.outer(err_z, err_z)
            Pxz += Wc[i] * np.outer(err, err_z)
        Pzz += R

        K = Pxz @ np.linalg.pinv(Pzz)
        err = measurements - z_pred
        err = self.normalize_angle(err)
        self.state = self.state + K @ err
        self.state = self.normalize_angle(self.state)
        self.P = self.P - K @ Pzz @ K.T
        return sigma_points_, z_pred, Pzz, Pxz, K
    
    def predict_and_update(self, time:float, measurements:np.ndarray, R:np.ndarray=None):
        sigma_points = self.predict(time)
        if measurements is not None:
            self.update(sigma_points, measurements, R)
        return self.state, self.P



    
def run_ekf_with_df(df:pd.DataFrame):
    # iterate over each object
    model = None
    states = []
    Ps = []
    
    # run kalman filter
    for index, row in df.iterrows():
        # initialize CTRA model
        if model is None:
            model = CTRAModel()
            model.init_state(row["time"], np.array([row["x"], row["y"], row["yaw"],  row["vx"], row["omega"], 0.0]).reshape(-1,1))
            # model = CTRVModel()
            # model.init_state(row["time"], np.array([row["x"], row["y"], row["yaw"], row["vx"], row["omega"]]).reshape(-1,1))
            states.append(model.state)
            Ps.append(model.P)
            continue

        # kf predict and update
        state, P = model.predict_and_update(row["time"], np.array([row["x"], row["y"], row["yaw"]]).reshape(-1,1))
        states.append(state)
        Ps.append(P)

    kfs_states = [None] * len(states)
    kfs_Ps = [None] * len(states)

    kfs_states[-1] = states[-1]
    kfs_Ps[-1] = Ps[-1]
    # backward smoothing
    for i in reversed(range(len(states)-1)):
        F = model.get_jacobian_F(states[i+1], df["time"].iloc[i+1] - df["time"].iloc[i])
        Fforward = model.get_jacobian_F(states[i], df["time"].iloc[i+1] - df["time"].iloc[i])
        x_hat = model.state_transition(states[i], df["time"].iloc[i+1] - df["time"].iloc[i])
        P_hat = Fforward @ Ps[i] @ Fforward.T + model.Q
        C = Ps[i] @ F.T @ np.linalg.inv(P_hat)
        kfs_states[i] = states[i] + C @ (kfs_states[i+1] - x_hat)
        kfs_Ps[i] = Ps[i] + C @ (kfs_Ps[i+1] - P_hat) @ C.T


    # list to df
    states = np.array(states).squeeze()
    kf_df = pd.DataFrame(states, columns=["x", "y", "yaw", "vx", "omega", "ax"], index=df.index)
    # kf_df = pd.DataFrame(states, columns=["x", "y", "yaw", "vx", "omega"], index=df.index)
    kf_df["time"] = df["time"].copy()

    kfs_states = np.array(kfs_states).squeeze()
    kfs_df = pd.DataFrame(kfs_states, columns=["x", "y", "yaw", "vx", "omega", "ax"], index=df.index)
    # kfs_df = pd.DataFrame(kfs_states, columns=["x", "y", "yaw", "vx", "omega"], index=df.index)
    kfs_df["time"] = df["time"].copy()
    return kf_df, kfs_df
    
import ipywidgets as widgets
from IPython.display import display
def interactive_plot(car_df, unique_ids):
    def plot_column(column_name):
        plt.figure(figsize=(10, 6))
        for id in unique_ids:
            selected_df = car_df[car_df["uuid"] == id]
            new_df = run_ekf_with_df(selected_df)
            plt.plot(selected_df["time"], selected_df[column_name], '.', label=f"raw {column_name}")
            plt.plot(new_df["time"], new_df[column_name], '--', label=f"smoothed {column_name}")
        plt.legend()
        plt.grid()
        plt.show()

    # 列名を選択するためのドロップダウンウィジェットを作成
    column_selector = widgets.Dropdown(
        options=car_df.columns,  # DataFrameの列名
        value='yaw',  # 初期値
        description='Column:',  # ウィジェットの説明
        disabled=False,
    )
    # ウィジェットとプロット関数を連携
    widgets.interact(plot_column, column_name=column_selector)

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


    df = load_rosbag_data(rosbag_file_path, topics)
    print(df["classification"].unique())
    car_labels = [ObjectClassification.CAR, ObjectClassification.TRUCK, ObjectClassification.BUS, ObjectClassification.TRAILER]
    car_df = df[df["classification"].isin(car_labels)]
    unique_ids = car_df["uuid"].unique()
    # interactive_plot(car_df, unique_ids)
    for id in unique_ids:
        selected_df = car_df[car_df["uuid"] == id]
        kf_df, kfs_df = run_ekf_with_df(selected_df)
        plt.figure()
        plt.title(f"uuid: {id}, x, yaw, vx, omega")
        plt.subplot(4,1,1)
        plt.plot(selected_df["time"], selected_df["x"],'.', label="raw")
        plt.plot(kf_df["time"], kf_df["x"], '--',label="ekf")
        plt.plot(kfs_df["time"], kfs_df["x"], '-.',label="smoothed")
        # plt.plot(kf_df2["time"], kf_df2["x"], '-.-',label="ukf")
        plt.legend()
        plt.grid()
        plt.title("x")
        plt.subplot(4,1,2)
        plt.plot(selected_df["time"], selected_df["yaw"],'.', label="raw")
        plt.plot(kf_df["time"], kf_df["yaw"], '--',label="ekf")
        plt.plot(kfs_df["time"], kfs_df["yaw"], '-.',label="smoothed")
        # plt.plot(kf_df2["time"], kf_df2["yaw"], '-.-',label="ukf")
        plt.legend()
        plt.grid()
        plt.title("yaw")
        plt.subplot(4,1,3)
        plt.plot(selected_df["time"], selected_df["vx"],'.', label="raw")
        plt.plot(kf_df["time"], kf_df["vx"], '--',label="ekf")
        plt.plot(kfs_df["time"], kfs_df["vx"], '-.',label="smoothed")
        # plt.plot(kf_df2["time"], kf_df2["vx"], '-.-',label="ukf")
        plt.legend()
        plt.grid()
        plt.title("vx")
        plt.subplot(4,1,4)
        plt.plot(selected_df["time"], selected_df["omega"],'.', label="raw")
        plt.plot(kf_df["time"], kf_df["omega"], '--',label="kf")
        plt.plot(kfs_df["time"], kfs_df["omega"], '-.',label="smoothed")
        plt.legend()
        plt.grid()
        plt.title("omega")
    plt.show()


if __name__ == "__main__":
    main()
