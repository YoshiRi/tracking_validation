# plotly visualizer
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import math
import random

# selecter
import tkinter as tk
from tkinter import filedialog

# data parser
from tracking_parser import DetctionAndTrackingParser
from utils import *

class DataForVisualize:
    def __init__(self, timestamp:float = 0.0, classification: int = 0, 
                 x:float = 0.0, y:float = 0.0, yaw:float = 0.0, length:float = 2.0, width:float = 1.0,
                 topic_name:str = "default", uuid = None):
        self.timestamp = timestamp
        self.classification = classification
        self.x = x
        self.y = y
        self.yaw = yaw
        self.length = length
        self.width = width
        self.topic_name = topic_name
        self.uuid = uuid
    
    def setTopicName(self, topic_name):
        self.topic_name = topic_name

    def fromPerceptionObjectsWithTime(self, perception_object, time):
        self.timestamp = time
        self.classification = getLabel(perception_object)
        self.x, self.y =  get2DPosition(perception_object)
        self.yaw = getYaw(perception_object)
        self.length = perception_object.shape.dimensions.x
        self.width = perception_object.shape.dimensions.y

    def setRandomState(self):
        self.x = random.random() * 10
        self.y = random.random() * 10
        self.yaw = random.random() * 2 * math.pi


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
    def __init__(self, rosbag_file_path: str = ""):
        # load rosbag data
        if rosbag_file_path == "":
            # select rosbag file from tkinter gui
            rosbag_file_path = self.select_file_gui()
        
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
        self.app = dash.Dash(__name__)
        self.app.layout = html.Div([
            dcc.Graph(id='graph', style={'height': '80vh'}),
            html.Label('Select timestamp range:'),
            dcc.RangeSlider(
                id='slider',
                min=min(obj.timestamp for obj in self.data_list),
                max=max(obj.timestamp for obj in self.data_list),
                value=[min(obj.timestamp for obj in self.data_list), max(obj.timestamp for obj in self.data_list)],
                marks={i: '{}'.format(i) for i in range(int(min(obj.timestamp for obj in self.data_list)), int(max(obj.timestamp for obj in self.data_list))+1)},
                step=None
            ),
            html.Label('visualization topic:'),
            dcc.Checklist(
                id='topic-checkbox',
                options=[{'label': i, 'value': i} for i in set(obj.topic_name for obj in self.data_list)],
                value=list(set(obj.topic_name for obj in self.data_list)),
                inline=True
            ),
            html.Label('visualization class:'),
            dcc.Checklist(
                id='class-checkbox',
                options=[{'label': 'Class {}'.format(i), 'value': i} for i in set(obj.classification for obj in self.data_list)],
                value=list(set(obj.classification for obj in self.data_list)),
                inline=True
            ),
        ], style={'height': '100vh'})

    def set_dash_app_callback(self):
        # set callback
        @self.app.callback(
        Output('graph', 'figure'),
        [Input('slider', 'value'),
        Input('topic-checkbox', 'value'), 
        Input('class-checkbox', 'value')]
        )
        def update_figure(selected_time_range, selected_topics, selected_classes):
            traces = []
            for obj in self.data_list:
                if selected_time_range[0] <= obj.timestamp <= selected_time_range[1] and obj.classification in selected_classes and obj.topic_name in selected_topics:
                    x_series, y_series = generate_rectangle_xy_series(obj.x, obj.y, obj.yaw, obj.length, obj.width)

                    traces.append(go.Scatter(
                        x=x_series,
                        y=y_series,
                        mode="lines",
                        line=dict(color=class_color_map[obj.classification], width=1),
                        fill="none",
                        hoverinfo="none",
                        showlegend=False
                    ))
                    
            return {'data': traces, 'layout': go.Layout(yaxis=dict(scaleanchor='x',), )}


    def load_rosbag_data(self, rosbag_file_path: str):
        # create self.data_list
        if rosbag_file_path == "dummy":
            # create dummy data
            self.data_list = []
            for i in range(5):
                viz_data = DataForVisualize()
                viz_data.setRandomState()
                viz_data.classification = i
                self.data_list.append(viz_data)
            return
        # parser is written in tracking_parser
        parser = DetctionAndTrackingParser(rosbag_file_path)
        # show detection only for now
        # topic_set is list of [time, DetectedObjects]
        topic_set = parser.data["/perception/object_recognition/detection/objects"]
        self.data_list = []
        for topic in topic_set:
            time = topic[0]
            obj = topic[1]
            viz_data = DataForVisualize()
            viz_data.fromPerceptionObjectsWithTime(time, obj)

            self.data_list.append(viz_data)

    def run_server(self):
        self.app.run_server(debug=True)





if __name__ == '__main__':
    #app.run_server(debug=True)
    fname = "dummy"
    visualizer = object2DVisualizer(fname)
    visualizer.run_server()
