# plotly visualizer
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import math
import random
import pandas as pd
import numpy as np

# selecter
import tkinter as tk
from tkinter import filedialog

# data parser
from tracking_parser import DetctionAndTrackingParser
from utils import *
from autoware_auto_perception_msgs.msg import ObjectClassification 
# object class label: ex) ObjectClassification.BUS == int(3)

class DataForVisualize:
    def __init__(self, timestamp:float = 0.0, classification: int = 0, 
                 x:float = 0.0, y:float = 0.0, yaw:float = 0.0, length:float = 2.0, width:float = 1.0,
                 topic_name:str = "default", uuid = None):
        self.data = {}
        self.data["time"] = timestamp
        self.data["classification"] = classification
        self.data["x"] = x
        self.data["y"] = y
        self.data["yaw"] = yaw
        self.data["length"] = length
        self.data["width"] = width
        self.data["topic_name"] = topic_name
        #self.data["uuid"] = uuid # currently uuid is not used in this class
    
    def setTopicName(self, topic_name):
        self.data["topic_name"] = topic_name

    def fromPerceptionObjectsWithTime(self, perception_object, time):
        self.data["time"] = time
        self.data["classification"] = getLabel(perception_object)
        self.data["x"], self.data["y"] =  get2DPosition(perception_object)
        self.data["yaw"] = getYaw(perception_object)
        self.data["length"] = perception_object.shape.dimensions.x
        self.data["width"] = perception_object.shape.dimensions.y

    def setRandomState(self):
        self.data["time"] = random.random() * 10
        self.data["classification"] = random.randint(0, 5)
        self.data["x"] = random.random() * 10
        self.data["y"] = random.random() * 10
        self.data["yaw"] = random.random() * 2 * math.pi
        self.data["length"] = 5
        self.data["width"] = 2



class_color_map = {
    0: "rgb(200, 200, 200)", # unkown
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



def generate_rectangle_xy_series(x, y, yaw, length, width):
    # Calculate the corner points of the rectangle
    theta = yaw
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    half_length = length / 2
    half_width = width / 2

    x0 = x - half_length * cos_theta - half_width * sin_theta
    y0 = y - half_length * sin_theta + half_width * cos_theta
    x1 = x + half_length * cos_theta - half_width * sin_theta
    y1 = y + half_length * sin_theta + half_width * cos_theta
    x2 = x + half_length * cos_theta + half_width * sin_theta
    y2 = y + half_length * sin_theta - half_width * cos_theta
    x3 = x - half_length * cos_theta + half_width * sin_theta
    y3 = y - half_length * sin_theta - half_width * cos_theta
    return [x0, x1, x2, x3, x0], [y0, y1, y2, y3, y0]

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
    
    def select_file_gui(self):
        # select rosbag file from tkinter gui
        root = tk.Tk()
        root.withdraw()  # do not show tkinter window
        file_path = filedialog.askopenfilename()  # get full file path
        return file_path

    def init_dash_app(self):
        """decide layout and set callback
        """
        self.set_dash_app_layout()
        self.set_dash_app_callback()

    def set_dash_app_layout(self):
        # config data
        timestamps = self.df["time"].to_list()
        min_time = min(timestamps)
        max_time = max(timestamps)
        int_min_time = math.floor(min_time)
        int_max_time = math.floor(max_time) + 1
        unique_topics = self.df["topic_name"].unique()
        unique_classes = self.df["classification"].unique()

        self.app = dash.Dash(__name__)
        self.app.layout = html.Div([
            dcc.Graph(id='graph', style={'height': '80vh'}),
            html.Label('Select timestamp[s] range:'),
            dcc.RangeSlider(
                id='slider',
                min=min_time,
                max=max_time,
                value=[min_time, max_time],
                #marks={i: '{}'.format(i) for i in range(int_min_time, int_max_time+1)},
                marks={i: '{}'.format(i) for i in timestamps},
                step=None
            ),
            html.Label('visualization topic:'),
            dcc.Checklist(
                id='topic-checkbox',
                options=[{'label': i, 'value': i} for i in unique_topics],
                value=unique_topics,
                inline=True
            ),
            html.Label('visualization class:'),
            dcc.Checklist(
                id='class-checkbox',
                options=[{'label': class_name_map[i], 'value': i} for i in unique_classes],
                value=unique_classes,
                inline=True
            ),
        ], style={'height': '100vh'})


    def set_dash_app_callback(self):
        # set callback
        @self.app.callback(
        Output('graph', 'figure'),
        [Input('slider', 'value'),
        Input('topic-checkbox', 'value'), 
        Input('class-checkbox', 'value'),
        State('graph', 'figure')
        ]
        )
        def update_figure(selected_time_range, selected_topics, selected_classes, figure):

            traces = []
            # filter df by selected time range
            time_range_condition = (self.df["time"] >= selected_time_range[0]) & (self.df["time"] <= selected_time_range[1])
            class_condition = self.df["classification"].isin(selected_classes)
            topic_condition = self.df["topic_name"].isin(selected_topics)
            df_filtered = self.df[time_range_condition & class_condition & topic_condition]

            # keep layout
            if figure is None:
                layout = go.Layout(yaxis=dict(scaleanchor='x',), )
            else:
                layout = figure["layout"]

            # generate traces
            for index, row in df_filtered.iterrows():
                x_series, y_series = generate_rectangle_xy_series(row["x"], row["y"], row["yaw"], row["length"], row["width"])
                traces.append(go.Scatter(
                    x=x_series,
                    y=y_series,
                    mode="lines",
                    line=dict(color=class_color_map[row["classification"]], width=1),
                    fill="none",
                    hoverinfo="none",
                    showlegend=False
                ))
                    
            return {'data': traces, 'layout': layout}

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
                data_dict = {}
                time = topic[0]*1e-9 # ns to sec
                obj = topic[1]
                viz_data = DataForVisualize()
                viz_data.fromPerceptionObjectsWithTime(obj, time)
                viz_data.setTopicName(topic_name)
                temp_list.append(viz_data.data)
        self.df = pd.DataFrame(temp_list)


    def run_server(self):
        # do not reload
        self.app.run_server(debug=True, use_reloader=False)
        print("quit dash server!")



def main(rosbag_file_path: str = "", topics: list = []):
    """main function to run visualizer

    Args:
        rosbag_file_path (str, optional): _description_. Defaults to "".
    """
            # create dummy data
    if rosbag_file_path == "":
        rosbag_file_path = "dummy"
        rosbag_file_path = "/home/yoshiri/autoware/inittest/rosbag2_2023_05_09-09_24_48_0.db3"

    visualizer = object2DVisualizer(rosbag_file_path, topics)
    visualizer.run_server()



if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rosbag", type=str, help="rosbag file path", default="")
    p.add_argument("--topics", nargs='+', help='List of strings', default=[])

    args = p.parse_args()
    rosbag_file_path = args.rosbag
    topics = args.topics
    # rosbag_file_path = "dummy"
    main(rosbag_file_path, topics)