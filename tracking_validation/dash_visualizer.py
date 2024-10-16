# plotly visualizer
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import random
import pandas as pd
import numpy as np

# selecter
import tkinter as tk
from tkinter import filedialog
import os

# data parser
from tracking_parser import DetctionAndTrackingParser
from utils import *
from autoware_auto_perception_msgs.msg import ObjectClassification 
# object class label: ex) ObjectClassification.BUS == int(3)

from typing import Dict, List, Tuple, Union

# dash colors
DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
            'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
            'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
            'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
            'rgb(188, 189, 34)', 'rgb(255, 255, 0)']

# symbols: circle, square, diamond, cross, x, triangle, pentagon, hexagram, star, diamond, hourglass, bowtie, asterisk, hash
DEFAULT_MARKER_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle', 'pentagon', 'hexagram', 'star', 'diamond', 'hourglass', 'bowtie', 'asterisk', 'hash']

class DataForVisualize:
    def __init__(self, timestamp:float = 0.0, classification: int = 0, 
                 x:float = 0.0, y:float = 0.0, yaw:float = 0.0, length:float = 2.0, width:float = 1.0,
                 topic_name:str = "default", uuid = []):
        self.data = {}
        self.data["time"] = timestamp
        self.data["classification"] = classification
        self.data["x"] = x
        self.data["y"] = y
        self.data["yaw"] = yaw
        self.data["length"] = length
        self.data["width"] = width
        self.data["topic_name"] = topic_name
        self.data["uuid"] = uuid
        self.data["cov_x"] = 0.0
        self.data["cov_y"] = 0.0
        self.data["cov_yaw"] = 0.0
        self.data["cov_vx"] = 0.0
        self.data["cov_omega"] = 0.0
        self.refine_uuid()
    
    def setTopicName(self, topic_name):
        self.data["topic_name"] = topic_name

    def refine_uuid(self):
        self.data["uuid"] = tuple(self.data["uuid"])

    def fromPerceptionObjectsWithTime(self, perception_object, time):
        self.data["time"] = time
        self.data["classification"] = getLabel(perception_object)
        self.data["x"], self.data["y"] =  get2DPosition(perception_object)
        self.data["yaw"] = getYaw(perception_object)
        self.data["length"] = perception_object.shape.dimensions.x
        self.data["width"] = perception_object.shape.dimensions.y
        twist = getTwist(perception_object)
        self.data["vx"] = twist[0]
        self.data["omega"] = twist[5]
        self.data["cov_x"] = perception_object.kinematics.pose_with_covariance.covariance[0]
        self.data["cov_y"] = perception_object.kinematics.pose_with_covariance.covariance[7]
        self.data["cov_yaw"] = perception_object.kinematics.pose_with_covariance.covariance[35]
        self.data["cov_vx"] = perception_object.kinematics.twist_with_covariance.covariance[0]
        self.data["cov_omega"] = perception_object.kinematics.twist_with_covariance.covariance[35]
        if type(perception_object) == DetectedObject:
            self.data["uuid"] = []
        else:
            self.data["uuid"] = perception_object.object_id.uuid
        self.refine_uuid()

    def setRandomState(self):
        self.data["time"] = random.random() * 10
        self.data["classification"] = random.randint(0, 5)
        self.data["x"] = random.random() * 10
        self.data["y"] = random.random() * 10
        self.data["yaw"] = random.random() * 2 * math.pi
        self.data["length"] = 5
        self.data["width"] = 2



class_color_map = {
    0: "rgb(200, 200, 200)", # unknown
    1: "rgb(255, 0, 0)", 
    2: "rgb(0, 255, 0)",
    3: "rgb(0, 0, 255)",
    4: "rgb(255, 255, 0)",
    5: "rgb(255, 0, 255)",
    6: "rgb(0, 255, 255)",
    7: "rgb(255, 255, 255)",
    8: "rgb(0, 0, 0)",
    # Add more colors for additional classes...
}

class_name_map = {
    ObjectClassification.UNKNOWN: "Unknown",
    ObjectClassification.BUS: "Bus",
    ObjectClassification.CAR: "Car",
    ObjectClassification.BICYCLE: "Bicycle",
    ObjectClassification.MOTORCYCLE: "Motorcycle",
    ObjectClassification.TRUCK: "Truck",
    ObjectClassification.TRAILER: "Trailer",
    ObjectClassification.PEDESTRIAN: "Pedestrian"
}



def generate_rectangle_xy_series(x, y, yaw, length, width, flag:bool = False):
    # Calculate the corner points of the rectangle
    theta = yaw
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    half_length = length / 2
    half_width = width / 2

    x0 = x - half_length * cos_theta - half_width * sin_theta # 
    y0 = y - half_length * sin_theta + half_width * cos_theta
    x1 = x + half_length * cos_theta - half_width * sin_theta
    y1 = y + half_length * sin_theta + half_width * cos_theta
    x2 = x + half_length * cos_theta + half_width * sin_theta
    y2 = y + half_length * sin_theta - half_width * cos_theta
    x3 = x - half_length * cos_theta + half_width * sin_theta
    y3 = y - half_length * sin_theta - half_width * cos_theta
    x_series = [x0, x1, x2, x3, x0]
    y_series = [y0, y1, y2, y3, y0]

    # Add an orientation indicator if flag is True
    if flag:
        # Calculate the midpoint of the front side of the rectangle
        front_mid_x = (x1 + x2) / 2
        front_mid_y = (y1 + y2) / 2

        # Calculate a point slightly forward of the midpoint
        orientation_length = length * 0.2  # 20% of the length of the rectangle
        orientation_x = front_mid_x + orientation_length * cos_theta
        orientation_y = front_mid_y + orientation_length * sin_theta

        # Insert the orientation point into the series
        x_series.insert(2, orientation_x)
        y_series.insert(2, orientation_y)
    return x_series, y_series
class object2DVisualizer:
    """visualize object in 2D map with plotly, dash

    Returns:
        _type_: _description_
    """
    def __init__(self, rosbag_file_path: str = "", topic_names: list = []):
        # load rosbag data
        if rosbag_file_path == "":
            # select rosbag file from tkinter gui
            rosbag_file_path = self.select_file_gui()
        self.topic_names = topic_names

        # load rosbag data
        self.load_rosbag_data(rosbag_file_path)    
        # init dash app
        self.init_dash_app()
        self.fig = None
    
    def select_file_gui(self):
        if os.getenv('DISPLAY'):
            root = tk.Tk()
            root.withdraw()  # Do not show Tkinter window
            file_path = filedialog.askopenfilename()  # Get full file path
        else:
            file_path = input("Enter the path to the ROS bag file: ")
        return file_path
    
    def parse_contents(self, contents, filename):
        print("currently not supported")
        # I gave up to implement this function because with browser, it is difficult to read full path of local file and we need the full path to read the file.

    def init_dash_app(self):
        """decide layout and set callback
        """
        self.enable_xy_crop = True
        self.set_dash_app_layout()
        self.set_dash_app_callback()
        self.toggle_crop() # default crop is off after init

    def set_dash_app_layout(self):
        # config data
        timestamps = self.df["time"].to_list()
        min_time = min(timestamps)
        max_time = max(timestamps)
        max_x = max(self.df["x"].to_list())
        max_y = max(self.df["y"].to_list())
        min_x = min(self.df["x"].to_list())
        min_y = min(self.df["y"].to_list())
        unique_topics = self.df["topic_name"].unique()
        unique_classes = self.df["classification"].unique()
        object_type = ["bounding_box", "object_center"]
        timestamp_range = [0.05, 0.1, 0.5, 1, 2, 3, 5, 7, 10] # log scale for slider

        # search unique uuid
        unique_uuids = self.df["uuid"].unique()
        self.uuid_color_map = {}
        self.topic_marker_map = {}
        self.topic_color_map: Dict[int,str] = {}
        for uuid in unique_uuids:
            self.uuid_color_map[uuid] = DEFAULT_PLOTLY_COLORS[random.randint(0, len(DEFAULT_PLOTLY_COLORS)-1)]
        
        for iter, unique_topic in enumerate(unique_topics):
            symbol_num = len(DEFAULT_MARKER_SYMBOLS)
            self.topic_marker_map[unique_topic] = DEFAULT_MARKER_SYMBOLS[iter % symbol_num]
            self.topic_color_map[unique_topic] = DEFAULT_PLOTLY_COLORS[iter % len(DEFAULT_PLOTLY_COLORS)]

        self.app = Dash(__name__)
        self.app.layout = html.Div([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False  # 一度に1つのファイルのみアップロード
                        ),
                        html.Div(id='output-data-upload'),
                        dcc.Graph(id='graph', style={'height': '80vh'}, # 80% of vertical height
                        config={
                            'staticPlot': False,        # check if we can use static plot
                            'scrollZoom': True,         # zoom with scroll
                            'doubleClick': 'reset',     # reset zoom with double click
                            'showTips': True,           # show tips
                            'displayModeBar': True,     # display mode bar
                            'modeBarButtonsToRemove': ['zoom2d',  'lasso2d'],  # remove zoom and lasso
                            'modeBarButtonsToAdd': ['pan2d', 'select2d']  # add pan and select
                        }),
            html.Label('Select timestamp[s] to focus:'),
            dcc.Slider(
                id='slider',
                min=min_time,
                max=max_time,
                value=min_time,
                #marks={i: '{}'.format(i) for i in range(int_min_time, int_max_time+1)},
                marks={i: '{}'.format(i) for i in timestamps},
                step=None
            ),
            html.Label('Select timestamp[s] width range:'),
            dcc.Slider(
                id='range-slider',
                min=min(timestamp_range),
                max=max(timestamp_range),
                value=0.1,
                #marks={i: '{}'.format(i) for i in range(int_min_time, int_max_time+1)},
                marks={i: '{}'.format(i) for i in timestamp_range},
                step=None
            ),
            # toggle crop button
            html.Label('Crop: red is on, green is off'),
            html.Div([
                html.Button('Toggle Crop On/Off', id='crop-button', n_clicks=0),
                html.Button('Update Range', id='update-range-button', n_clicks=0)
            ]),

            html.Label('visualization topic:'),
            dcc.Checklist(
                id='topic-checkbox',
                options=[{'label': i, 'value': i} for i in unique_topics],
                value=[unique_topics[0]],
                inline=True
            ),
            html.Label('visualization class:'),
            dcc.Checklist(
                id='class-checkbox',
                options=[{'label': class_name_map[i], 'value': i} for i in unique_classes],
                value=unique_classes,
                inline=True
            ),
            html.Label('visualization mark:'),
            dcc.Checklist(
                id='object-mark-checkbox',
                options=[{'label': i, 'value': i} for i in object_type],
                value=[object_type[0]],
                inline=True
            ),
            html.Label('Plot color by:'),
            dcc.RadioItems(
                id='radio-items',
                options = ['topic name', 'class label'],
                value = 'class label',
                inline=True
            ),
            html.Label('Time series key:'),
            dcc.Dropdown(
                id='ts-dropdown',
                options=[
                    {'label': 'x', 'value': 'x'},
                    {'label': 'y', 'value': 'y'},
                    {'label': 'yaw', 'value': 'yaw'},
                    {'label': 'length', 'value': 'length'},
                    {'label': 'width', 'value': 'width'},
                    {'label': 'vx', 'value': 'vx'},
                    {'label': 'omega', 'value': 'omega'},
                    {'label': 'elapsed_time', 'value': 'elapsed_time'},
                    {'label': 'cov_x', 'value': 'cov_x'},
                    {'label': 'cov_y', 'value': 'cov_y'},
                    {'label': 'cov_yaw', 'value': 'cov_yaw'},
                    {'label': 'cov_vx', 'value': 'cov_vx'},
                    {'label': 'cov_omega', 'value': 'cov_omega'},
                ],
                value='yaw'  # デフォルトの値
            ),
        ], style={'height': '100vh'})

    def toggle_crop(self):
        #set callback
        @self.app.callback(
            Output('crop-button', 'style'),
            [Input('crop-button', 'n_clicks')]
        )
        def toggle_crop_callback(n_clicks):
            # do not use n_clicks
            self.enable_xy_crop = not self.enable_xy_crop # toggle
            if self.enable_xy_crop:
                return {'background-color': 'red'}
            else:
                return {'background-color': 'green'}

    def set_dash_app_callback(self):
        # upload callback
        @self.app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
        def update_output(contents, filename):
            if contents is not None:
                return self.parse_contents(contents, filename)

        # draw callback
        @self.app.callback(
        Output('graph', 'figure'),
        [Input('slider', 'value'),
        Input('range-slider', 'value'),
        Input('topic-checkbox', 'value'), 
        Input('class-checkbox', 'value'),
        Input('object-mark-checkbox', 'value'),
        Input('radio-items', 'value'),
        Input('update-range-button', 'n_clicks'),
        Input('ts-dropdown', 'value'),
        State('graph', 'figure')
        ]
        )
        def update_figure(selected_time, selected_time_range, selected_topics, selected_classes, selected_object_type, color_policy, update_crop, time_series_key,fig_dict: Dict):
            # filter df by selected time range
            time_range_condition = (self.df["time"] >= selected_time - selected_time_range) & (self.df["time"] <= selected_time + selected_time_range)
            class_condition = self.df["classification"].isin(selected_classes)
            topic_condition = self.df["topic_name"].isin(selected_topics)
            df_filtered = self.df[time_range_condition & class_condition & topic_condition]
            df_timeseries = self.df[class_condition & topic_condition]
            # if toggle is chosed, crop xy range
            if self.enable_xy_crop:
                try: # filter out when fig_dict is None or no range information
                    layout = fig_dict["layout"]
                    x_range = layout['xaxis']['range']
                    y_range = layout['yaxis']['range']
                    x_range_condition = (df_timeseries["x"] >= x_range[0]) & (df_timeseries["x"] <= x_range[1])
                    y_range_condition = (df_timeseries["y"] >= y_range[0]) & (df_timeseries["y"] <= y_range[1])
                    df_timeseries = df_timeseries[x_range_condition & y_range_condition]
                    # print(x_range, y_range)
                except Exception as e:
                    print(e)

            # first time to plot
            if fig_dict is None:
                layout = go.Layout(yaxis=dict(scaleanchor='x',), dragmode="pan", hovermode="closest")
                self.fig = make_subplots(rows=1, cols=2, shared_xaxes=False, shared_yaxes=False,
                                subplot_titles=('2D Plot', 'Time Series'))
            else:
                layout = fig_dict["layout"]
            
            # remove old traces in left 2d plot
            try:
                self.fig.data = []
            except Exception as e:
                print(e, "no data in fig. create new fig")
                layout = go.Layout(yaxis=dict(scaleanchor='x',), dragmode="pan", hovermode="closest")
                self.fig = make_subplots(rows=1, cols=2, shared_xaxes=False, shared_yaxes=False,
                                subplot_titles=('2D Plot', 'Time Series'))

            # update left 2d plot
            traces_2d = []
            if "bounding_box" in selected_object_type:
                traces_2d += self.plot_2d_bbox_traces(df_filtered, color_policy)
            if "object_center" in selected_object_type:
                traces_2d += self.plot_2d_center_traces(df_filtered, color_policy)
            for trace in traces_2d:
                self.fig.add_trace(trace, row=1, col=1)
            self.fig.update_layout(layout)

            # time series plot
            self.plot_time_series(self.fig, df_timeseries, selected_time, selected_time_range, color_policy, time_series_key )
            # update subplot title in right time series plot
            self.fig.update_layout(title_text="Time Series:" + time_series_key, title_x=0.5, title_y=0.9, title_font_size=20, title_font_family="Arial")
            self.fig.update_xaxes(range=[selected_time - selected_time_range, selected_time +  selected_time_range], row=1, col=2)
            self.fig.update_yaxes(range=[df_timeseries[time_series_key].min(), df_timeseries[time_series_key].max()], row=1, col=2)
            return self.fig
        
    def draw_vline(self, fig: go.Figure, x: float):
        """draw vertical line in figure

        Args:
            fig (go.Figure): _description_
            x (float): _description_
        """
        pass

    def plot_time_series(self, fig: go.Figure, df_filtered: pd.DataFrame, selected_time: float, selected_time_range: float, color_policy: str = "topic name", key:str = "x"):
        """plot time series

        Args:
            df_filtered (pd.DataFrame): _description_
            selected_time (float): _description_
            selected_time_range (float): _description_
        """
        # plot time series
        unique_topics = df_filtered["topic_name"].unique()
        for topic in unique_topics:
            tmp_df = df_filtered[df_filtered["topic_name"] == topic]
            unique_uuids = tmp_df["uuid"].unique()
            color = self.topic_color_map[topic]
            for uuid in unique_uuids:
                plot_df = tmp_df[tmp_df["uuid"] == uuid]
                draw_mode:str = "lines" if uuid else "markers"
                color = color if color_policy == "topic name" else self.uuid_color_map[uuid]
                # draw time series
                fig.add_trace(go.Scatter(
                    x=plot_df["time"],
                    y=plot_df[key],
                    mode=draw_mode,
                    line=dict(color=color, width=1),
                    hoverinfo="none",
                    showlegend=False
                ), row=1, col=2)

    # plot 2d 
    def plot_2d_bbox_traces(self, df_filtered: pd.DataFrame, color_policy: str = "topic name"):
        """plot 2d bounding box traces with plotly

        Args:
            df_filtered (pd.DataFrame): _description_
        Returns:
            2d_traces (list): list of traces
        """
        # generate traces
        traces = []
        for index, row in df_filtered.iterrows():
            if color_policy == "topic name":
                # color determined by topic name 
                color = self.topic_color_map[row["topic_name"]]
            elif color_policy == "class label":
                # color determined by class
                color = class_color_map[row["classification"]]
            else:
                raise NotImplementedError
            

            x_series, y_series = generate_rectangle_xy_series(row["x"], row["y"], row["yaw"], row["length"], row["width"], flag=True)
            traces.append(go.Scatter(
                x=x_series,
                y=y_series,
                mode="lines",
                line=dict(color=color, width=1),
                fill="none",
                hoverinfo="none",
                showlegend=False
            ))
        return traces
    
    def plot_2d_center_traces(self, df_filtered: pd.DataFrame, color_policy: str = "topic name"):
        """plot 2d center traces with plotly

        Args:
            df_filtered (pd.DataFrame): _description_
        Returns:
            2d_traces (list): list of traces
        """
        traces = []
        for index, row in df_filtered.iterrows():
            traces.append(go.Scatter(
                x=[row["x"]],
                y=[row["y"]],
                mode="markers",
                # line with marker with random color
                marker=dict(color=self.uuid_color_map[row["uuid"]], size=5, symbol=self.topic_marker_map[row["topic_name"]]),
                hoverinfo="none",
                #label=" ".join(str(x) for x in row["uuid"]),
                showlegend=False
            ))
        return traces


    def create_random_dummy_data(self, num: int):
        """create dummy data for testing

        Args:
            num (int): object number
        """
        data_list = []
        for i in range(num): # 1000 obj ok, 10000 obj ng
            viz_data = DataForVisualize()
            viz_data.setRandomState()
            data_list.append(viz_data.data)
        self.df = pd.DataFrame(data_list)


    def load_rosbag_data(self, rosbag_file_path: str):
        """load rosbag data and put data list to self.data_list

        Args:
            rosbag_file_path (str): _description_
        """
        # if name is dummy, set dummy data for debug
        if rosbag_file_path == "dummy":
            self.create_random_dummy_data(10)
            return
        
        # parser is written in tracking_parser
        parser = DetctionAndTrackingParser(rosbag_file_path, self.topic_names) # parser.data has dict of topic name and data

        self.object_to_df(parser.data)
        print("topic num: {}".format(len(self.df)))

    def object_to_df(self, data):
        """object to pandas data frame

        Args:
            data: dict with topic name keys
        """
        temp_list = []
        for topic_name in data:
            for topic in  data[topic_name]:
                # topic data is already parsed into list of perception object
                data_dict = {}
                time = topic[0]
                obj = topic[1]
                diff_time = topic[2]
                viz_data = DataForVisualize()
                viz_data.fromPerceptionObjectsWithTime(obj, time)
                viz_data.setTopicName(topic_name)
                viz_data.data["elapsed_time"] = diff_time
                temp_list.append(viz_data.data)
        self.df = pd.DataFrame(temp_list)


    def run_server(self):
        # do not reload
        self.app.run_server(debug=True, use_reloader=False, port=8051)
        print("quit dash server!")



def main(rosbag_file_path: str = "", topics: list = []):
    """main function to run visualizer

    Args:
        rosbag_file_path (str, optional): _description_. Defaults to "".
    """

    visualizer = object2DVisualizer(rosbag_file_path, topics)
    visualizer.run_server()



if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rosbag", type=str, help="rosbag file path", default="dummy")
    p.add_argument("--topics", nargs='+', help='List of strings', default=[])

    args = p.parse_args()
    rosbag_file_path = args.rosbag
    topics = args.topics
    if topics == []:
        # causion: too much topics make visualization slow.
        #          we recommend to comment out topics you don't need
        topics = ["/perception/object_recognition/detection/objects", "/perception/object_recognition/tracking/objects","/perception/object_recognition/tracking/near_objects", 
                  "/perception/object_recognition/detection/clustering/camera_lidar_fusion/objects", "/perception/object_recognition/detection/detection_by_tracker/objects", 
                  "/perception/object_recognition/detection/pointpainting/validation/objects", "/perception/object_recognition/detection/centerpoint/validation/objects",
                  "/perception/object_recognition/detection/radar/far_objects", "/perception/object_recognition/tracking/radar/far_objects",
                  "/tmp/perception/object_recognition/detection/objects", "/tmp/perception/object_recognition/tracking/objects","/tmp/perception/object_recognition/tracking/near_objects",
                    # "/tmp/perception/object_recognition/detection/clustering/camera_lidar_fusion/objects", "/tmp/perception/object_recognition/detection/detection_by_tracker/objects",
                    "/tmp/perception/object_recognition/detection/pointpainting/validation/objects", "/tmp/perception/object_recognition/detection/centerpoint/validation/objects",
                    # "/tmp/perception/object_recognition/detection/radar/far_objects", "/tmp/perception/object_recognition/tracking/radar/far_objects",
                  ]
    # rosbag_file_path = "dummy"
    main(rosbag_file_path, topics)