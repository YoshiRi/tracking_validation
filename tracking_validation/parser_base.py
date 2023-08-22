from typing import List, Dict, Tuple, Union, Optional
import pandas as pd
import matplotlib.pyplot as plt

class BaseParser:
    def __init__(self):
        # assume there are data frame named "df" and "filt_df"
        pass

    def dict_to_dataframe(self, data: Dict) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement this method")

    def plot_kinematics(self, df: pd.DataFrame, x_key: str, y_key: str, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        plt_style = kwargs.pop("plt_style", "x-")
        for id in df["id"].unique():
            data = df[df["id"] == id]
            ax.plot(data[x_key], data[y_key], plt_style)
        return ax

    def plot_data(self, x_key: str = "time", y_keys: List[str] = [], **kwargs):
        # Extracting the implementation from the original TrackingParser class
        if len(y_keys) == 0:
            y_keys = ["x", "y", "yaw", "vx", "covariance_x", "covariance_vx","length", "width"]
        
        # extract legend form
        # set original legend if uuid_legend is false
        uuid_legend = kwargs.pop("uuid_legend", False)
        if uuid_legend:
            legend = self.filt_df["id"].unique()
        else:
            legend = ["track_" + str(i) for i in range(len(self.filt_df["id"].unique()))]

        cols = int(kwargs.pop("cols", 2))
        rows = len(y_keys)//cols + len(y_keys)%cols
        figsize = (8*cols, 5 * rows)
        fig, axs = plt.subplots(rows, cols, sharex=True, figsize=figsize)
        axs = axs.reshape(-1)
        for i, y_key in enumerate(y_keys):
            self.plot_kinematics(self.filt_df, x_key, y_key, ax=axs[i], **kwargs)   
            axs[i].set_title(y_key)
            axs[i].set_xlabel("time [s]")
            axs[i].grid()
            axs[i].legend(legend)
        return fig, axs
    
    def plot_state_and_cov(self, **kwargs):
        x_key = "time"
        y_keys = ["x", "y", "yaw", "vx", "vyaw", "estimated sin(slip_angle)", "length", "width"]
        y_cov_keys = ["covariance_x", "covariance_y", "covariance_yaw", "covariance_vx", "covariance_vyaw", "existence_probability"]
        self.plot_data(x_key, y_keys, **kwargs)
        self.plot_data(x_key, y_cov_keys, **kwargs)

    def plot2d(self, **kwargs):
        uuid_legend = kwargs.pop("uuid_legend", False)
        if uuid_legend:
            legend = self.filt_df["id"].unique()
        else:
            legend = ["track_" + str(i) for i in range(len(self.filt_df["id"].unique()))]
        plt.figure(figsize=(12,12))
        ax = plt.gca()
        self.plot_kinematics(self.filt_df, "x", "y", plt_style="x-", ax=ax)
        plt.title("2D Position")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid()
        plt.legend(legend)

    def filter_df_between(self, dict_key, bound1, bound2):
        upper_bound = max(bound1, bound2)
        lower_bound = min(bound1, bound2)
        self.filt_df = self.filt_df[(self.filt_df[dict_key] < upper_bound) & (self.filt_df[dict_key] > lower_bound)]

    def crop_df_by_time(self, start_time, end_time):
        upper_bound = max(start_time, end_time)
        lower_bound = min(start_time, end_time)
        start = self.filt_df["time"].min()
        self.filt_df = self.filt_df[(self.filt_df["time"] < start + upper_bound) & (self.filt_df["time"] > start + lower_bound)]
    
    def filter_df_equal(self, dict_key, value):
        self.filt_df = self.filt_df[self.filt_df[dict_key] == value]

    def filter_df_by_label(self, labels):
        self.filt_df = self.filt_df[self.filt_df["class_label"].isin(labels)]
    
    def filter_reset(self):
        self.filt_df = self.df.copy()

    def filter_df_by_data_length(self, min_length: int):
        if min_length < 1:
            # do nothing
            return
        # for each object id in dataframe
        for id in self.filt_df["id"].unique():
            # get data length of each object id
            data_length = len(self.filt_df[self.filt_df["id"] == id])
            # if data length is less than min_length, remove data
            if data_length < min_length:
                self.filt_df = self.filt_df[self.filt_df["id"] != id]