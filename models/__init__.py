# For relative imports to work in Python 3.6
# import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from .BaseDetectionModel import BaseDetectionModel
from .YoloDetectionModel import YoloDetectionModel
from .SsdDetectionModel import SsdDetectionModel
from .FaceDetectionModel import FaceDetectionModel
from .CustomDetectionModel import CustomDetectionModel