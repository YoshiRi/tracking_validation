import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import math


import tkinter as tk
from tkinter import filedialog



class data:
    def __init__(self, timestamp:float, classification: int, 
                 x:float, y:float, yaw:float, length:float, width:float):
        self.timestamp = timestamp
        self.classification = classification
        self.x = x
        self.y = y
        self.yaw = yaw
        self.length = length
        self.width = width

class_color_map = {
    1: "rgb(255, 0, 0)",
    2: "rgb(0, 255, 0)",
    # Add more colors for additional classes...
}

class object2DVisualizer():
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
                value=[min(obj.timestamp for obj in data_list), max(obj.timestamp for obj in self.data_list)],
                marks={i: '{}'.format(i) for i in range(int(min(obj.timestamp for obj in self.data_list)), int(max(obj.timestamp for obj in self.data_list))+1)},
                step=None
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
        Input('class-checkbox', 'value')]
        )
        def update_figure(selected_time_range, selected_classes):
            traces = []
            for obj in self.data_list:
                if selected_time_range[0] <= obj.timestamp <= selected_time_range[1] and obj.classification in selected_classes:
                    x = obj.x
                    y = obj.y
                    length = obj.length
                    width = obj.width
                    yaw = obj.yaw

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

                    traces.append(go.Scatter(
                        x=[x0, x1, x2, x3, x0],
                        y=[y0, y1, y2, y3, y0],
                        mode="lines",
                        line=dict(color=class_color_map[obj.classification], width=1),
                        fill="toself",
                        hoverinfo="none",
                        showlegend=False
                    ))
                    
            return {'data': traces, 'layout': go.Layout(yaxis=dict(scaleanchor='x',), )}


    def load_rosbag_data(self, rosbag_file_path: str):
        #set dummy data
        self.data_list = [
            data(timestamp=1.0, classification=1, x=0.0, y=0.0, yaw=0.0, length=2.0, width=1.0),
            data(timestamp=2.0, classification=2, x=1.0, y=1.0, yaw=1.0, length=3.0, width=2.0),
            data(timestamp=3.0, classification=1, x=2.0, y=2.0, yaw=0.5, length=2.5, width=1.5),
            # additional data
        ]
        pass

    def run_server(self):
        self.app.run_server(debug=True)

if __name__ == '__main__':
    #app.run_server(debug=True)
    fname = "hoge"
    visualizer = object2DVisualizer(fname)
    visualizer.run_server()
