"""load association matrix params from ros parameters yaml file
"""

import numpy as np
import yaml
import os

DEFAULT_YAML_FILE = "$HOME/autoware/src/universe/autoware.universe/perception/multi_object_tracker/config/data_association_matrix.param.yaml"


def load_association_params(filename):
    """
    Loads the association matrix parameters from a yaml file.
    """

    # check if file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError("File not found: {}".format(filename))

    with open(filename, "r") as f:
        association_params = yaml.safe_load(f)["/**"][
            "ros__parameters"
        ]
    return association_params    


class AssociationParams:
    """class to handle association parameters
    """

    # keys for association matrix params
    #UNKNOWN,  CAR,      TRUCK,    BUS,      TRAILER,   MOTORBIKE, BICYCLE,  PEDESTRIAN
    #0,        1,        2,        3,        4,         5,         6,        7
    object_type = {
        "UNKNOWN": 0,
        "CAR": 1,
        "TRUCK": 2,
        "BUS": 3,
        "TRAILER": 4,
        "MOTORBIKE": 5,
        "BICYCLE": 6,
        "PEDESTRIAN": 7
    }

    # parameter keys:
    #   max_dist_matrix, min_area_matrix, max_area_matrix, max_rad_matrix, min_iou_matrix
    param_keys = ["max_dist_matrix", "min_area_matrix", "max_area_matrix", "max_rad_matrix", "min_iou_matrix"]

    def __init__(self, filename=DEFAULT_YAML_FILE):
        self.params = load_association_params(filename)

    def getAssociationParams(self, type1:str, type2:str):
        """get association params for two types
        """
        index1 = self.object_type[type1]
        index2 = self.object_type[type2]
        index = index1 * len(self.object_type.keys()) + index2

        params = []
        for key in self.param_keys:
            params.append(self.params[key][index])

        return params
