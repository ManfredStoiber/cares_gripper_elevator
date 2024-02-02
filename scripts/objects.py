import logging
import random
from enum import Enum
from time import sleep
import dynamixel_sdk as dxl
from functools import wraps

from serial import Serial

from cares_lib.vision.Camera import Camera
from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.vision.STagDetector import STagDetector

from cares_lib.dynamixel.Servo import Servo


def exception_handler(error_message):
    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            try:
                return function(self, *args, **kwargs)
            except EnvironmentError as error:
                logging.error(f"Environment for Gripper#{error.gripper.gripper_id}: {error_message}")
                raise EnvironmentError(error.gripper, f"Environment for Gripper#{error.gripper.gripper_id}: {error_message}") from error
        return wrapper
    return decorator


class Command(Enum):
    GET_YAW = 0
    OFFSET = 1

class ArucoObject(object):
    def __init__(self, camera: Camera, aruco_detector: ArucoDetector or STagDetector, object_marker_id: int) -> None:
        self.camera = camera
        self.aruco_detector = aruco_detector
        self.object_marker_id = object_marker_id

    def get_yaw(self, blindable=False, detection_attempts=10):
        attempt = 0
        while not blindable or attempt < detection_attempts:
            attempt += 1
            msg = f"{attempt}/{detection_attempts}" if blindable else f"{attempt}"
            logging.debug(f"Attempting to detect aruco target: {self.object_marker_id}")

            frame = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix,
                                                                self.camera.camera_distortion, display=False)
            if self.object_marker_id in marker_poses:
                return marker_poses[self.object_marker_id]["orientation"][2]
        return None

    def reset(self):
        pass

    def reset_target_servo(self):
        pass