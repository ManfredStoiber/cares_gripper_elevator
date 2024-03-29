import matplotlib.pyplot as plt
from cares_lib.vision.Camera import Camera
from cares_lib.vision.STagDetector import STagDetector
from Gripper import Gripper
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from Servo import Servo
from configurations import EnvironmentConfig, ObjectConfig
from abc import ABC, abstractmethod
from objects import ArucoObject
import dynamixel_sdk as dxl

import logging
import random
import time
from functools import wraps
from scipy.stats import trim_mean
from pathlib import Path
from enum import Enum
import numpy as np
import cv2


file_path = Path(__file__).parent.resolve()

VALVE_SERVO_ID = 10
SLEEP_TIME = 0.5  # in seconds


def exception_handler(error_message):
    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            try:
                return function(self, *args, **kwargs)
            except EnvironmentError as error:
                logging.error(f"Environment for Gripper#{error.gripper.gripper_id}: {error_message}")
                raise EnvironmentError(error.gripper,
                                       f"Environment for Gripper#{error.gripper.gripper_id}: {error_message}") from error

        return wrapper

    return decorator


class EnvironmentError(IOError):
    def __init__(self, gripper, message):
        self.gripper = gripper
        super().__init__(message)


class OBSERVATION_TYPE(Enum):
    SERVO = 0
    ARUCO = 1
    SERVO_ARUCO = 2
    IMAGE = 3


class Environment(ABC):
    """
    Initialise the environment with the given configurations of the gripper and object.

    Parameters:
    env_config: Configuration specific to the environment setup.
    gripper_config: Configuration specific to the gripper used.
    """

    def __init__(self, env_config: EnvironmentConfig, gripper_config: GripperConfig, object_config: ObjectConfig):

        self.current_color = 1

        self.gripper_config = gripper_config
        self.object_config = object_config
        self.gripper = Gripper(gripper_config)
        self.camera = Camera(env_config.camera_id, env_config.camera_matrix, env_config.camera_distortion)

        self.object_marker_id = object_config.object_marker_id
        self.object_observation_mode = object_config.object_observation_mode
        self.object_device_name = object_config.device_name
        self.object_baudrate = object_config.baudrate

        self.init_elevator()

        self.observation_type = env_config.observation_type
        self.action_type = gripper_config.action_type
        self.servo_type = gripper_config.servo_type

        self.blindable = env_config.blindable
        self.env_type = env_config.env_type

        self.goal_selection_method = env_config.goal_selection_method
        self.noise_tolerance = env_config.noise_tolerance

        self.marker_detector = STagDetector(marker_size=env_config.marker_size)
        self._return_to_home()

        aruco_yaw = None

        if self.object_observation_mode == "observed":
            aruco_yaws = []
            for i in range(0, 10):
                aruco_yaws.append(self.observed_object_state(marker_only=True)[2])
            aruco_yaw = trim_mean(aruco_yaws, 0.1)
        self.target = ArucoObject(
            self.camera, self.marker_detector, self.object_marker_id)

        self.object_state_before = self.goal_state = self.get_object_state()

    def restart_gripper(self):
        self.gripper = Gripper(self.gripper_config)
        self.init_elevator()
        self.change_LED_colors(self.current_color)

    def init_elevator(self):
        ## elevator
        self.elevator_port_handler = dxl.PortHandler(self.object_device_name)
        self.elevator_packet_handler = dxl.PacketHandler(2)
        self.elevator = Servo(self.elevator_port_handler, self.elevator_packet_handler, 2, 5,
                              1,
                              225, 225, 4096 * 3, - 4096 * 3,
                              "XL330-M077-T")

        if not self.elevator_port_handler.openPort():
            error_message = f"Failed to open port {self.object_device_name}"
            logging.error(error_message)
            raise IOError(error_message)
        logging.info(f"Succeeded to open port {self.object_device_name}")

        if not self.elevator_port_handler.setBaudRate(self.object_baudrate):
            error_message = f"Failed to change the baudrate to {self.object_baudrate}"
            logging.error(error_message)
            raise IOError(error_message)
        logging.info(f"Succeeded to change the baudrate to {self.object_baudrate}")
        self.elevator.enable()

    @exception_handler("Environment failed to reset")
    def reset(self):
        """
        Resets the environment for a new episode.

        This method wiggles the gripper to its home position, generates a random home
        position (angle) for the object, chooses a new goal angle (ensuring it's not
        too close to the home angle), and resets the target servo position if necessary.

        Returns:
        list: The initial state of the environment.
        """
        self._return_to_home()
        state = self.get_state()

        logging.debug(state)

        # choose goal will crash if not home
        self.goal_state = self.choose_goal()

        logging.info(f"New Goal Generated: {self.goal_state}")
        return state

    def _return_to_home(self):
        self.gripper.move([512, 512, 512, 512])  # close gripper

        # check if object between fingertips
        def is_object_between():
            try:
                marker_ids, marker_poses = self.get_marker_poses(max_attempts=100, max_attempts_object=100)
            except:
                return False
            object_state = self.get_object_state()
            return marker_poses[6]["position"][0] <= object_state[0] <= marker_poses[5]["position"][0]

        # reset until in default position
        while not (self.gripper.is_home() and is_object_between()):
            # reset gripper
            self.gripper.disable_torque()
            self.elevator.move_extended(4096 + 2500, wait=False)  # lift until little over ground
            time.sleep(1) # sleep instead of move_extended(.., timeout=1) to ensure waiting time
            self.elevator.move_extended(4096 + 5600, wait=False)  # lift only so high that it doesn't overshoot wall
            time.sleep(1)
            self.gripper.move([300, 512, 680, 512])  # open gripper
            self.elevator.move_extended(4096 + 2500, wait=False)  # put down until little over ground
            time.sleep(0.5)
            self.gripper.move([512, 512, 512, 512])  # close gripper
            self.elevator.move_extended(4096, wait=False)  # put down


    def sample_action(self):
        if self.action_type == "velocity":
            return self.sample_action_velocity()
        return self.sample_action_position()

    def sample_action_position(self):
        action = []
        for i in range(0, self.gripper.num_motors):
            min_value = self.gripper.min_values[i]
            max_value = self.gripper.max_values[i]
            action.append(random.randint(min_value, max_value))
        return action

    def sample_action_velocity(self):
        action = []
        for i in range(0, self.gripper.num_motors):
            action.append(random.randint(self.gripper.velocity_min, self.gripper.velocity_max))
        return action

    @exception_handler("Failed to step")
    def step(self, action):
        """
        Takes a step in the environment using the given action and returns the results.

        Parameters:
        action: The action to be executed.

        Returns:
        state: The new state after executing the action.
        reward: The reward obtained after the action.
        done: Whether the episode is done or not.
        truncated: Whether the step was truncated or not.
        """

        if self.action_type == "velocity":
            self.gripper.move_velocity_joint(action)
        else:
            self.gripper.move(action)

        state = self.get_state()
        logging.debug(f"New State: {state}")

        object_state_after = self.get_object_state()

        logging.debug(f"New Object State: {object_state_after}")

        reward, done = self.reward_function(self.goal_state, self.object_state_before, object_state_after)

        self.object_state_before = object_state_after

        truncated = False
        return state, reward, done, truncated

    @exception_handler("Failed to step gripper")
    def step_gripper(self):
        self.gripper.step()

    @exception_handler("Failed to get servo states")
    def servo_state_space(self):
        """
        Gets the current state of the environment when using servo observations.

        Returns:
        A list representing the state of the environment.
        """
        # Angle Servo + X-Y-Yaw Object + Goal
        state = []
        gripper_state = self.gripper.state()
        state += gripper_state["positions"]

        if self.action_type == "velocity":
            state += gripper_state["velocities"]
            if self.servo_type == "XL-320":
                state += gripper_state["loads"]

        state += self.get_object_state(yaw_only=False)

        state = self.add_goal(state)

        return state

    # The aruco state presumes the Aruco IDs match the servo IDs + a Marker for each finger tip + 1 Marker for Object
    @exception_handler("Failed to get aruco states")
    def aruco_state_space(self):
        """
        Gets the current state of the environment when using Aruco marker observations.

        Returns:
        A list representing the state of the environment.
        """
        # X-Y Servo + X-Y Finger-tips + X-Y-Yaw Object + Goal
        state = []

        while True:
            try:
                marker_ids, marker_poses = self.get_marker_poses()
                break
            except:
                self._return_to_home()

        # Add the XY poses for each of the markers in marker id into the state
        for id in marker_ids:
            marker_pose = marker_poses[id]
            position = marker_pose["position"]
            state.append(position[0])  # X
            state.append(position[1])  # Y

        if self.env_type == 0:
            # Add the additional yaw information from the object marker
            state += [marker_poses[self.object_marker_id]["orientation"][2]]  # Yaw

        state = self.add_goal(state)

        return state

    def get_marker_poses(self, max_attempts=np.inf, max_attempts_object=100):
        # Servos + Finger Tips (2) + Object (1)
        num_markers = self.gripper.num_motors + 3
        # maker_ids match servo ids (counting from 1)
        marker_ids = [id for id in range(1, num_markers + 1)]
        attempt = 0
        attempt_object = 0
        while True:
            if attempt > max_attempts:
                logging.warning(f"Could not determine all marker poses after {attempt} attempts")
                raise Exception(f"Could not determine all marker poses after {attempt} attempts")

            if attempt_object > max_attempts_object:
                logging.warning(f"Could not determine object marker pose after {attempt_object} attempts")
                raise Exception(f"Could not determine object marker pose after {attempt_object} attempts")

            attempt += 1
            logging.debug(f"Attempting to Detect State")
            frame = self.camera.get_frame()
            marker_poses = self.marker_detector.get_marker_poses(frame[:, :, [2, 1, 0]], self.camera.camera_matrix,
                                                                 self.camera.camera_distortion, display=False)

            # This will check that all the markers are detected correctly
            if all(ids in marker_poses for ids in marker_ids):
                break

            missing_markers = set(marker_ids).difference(set(marker_poses))

            if attempt % 10 == 1:  # only log every 10th attempt, starting from second
                logging.warning(f"Attempt {attempt} to detect markers failed. Missing marker(s): {missing_markers}")
                if attempt > 100:
                    cv2.imwrite(f"debugimage_{time.time()}.png", frame)

            if self.object_marker_id in missing_markers:
                attempt_object += 1
            else:
                attempt_object = 0



        return marker_ids, marker_poses

    @exception_handler("Failed to get servo and aruco states")
    def servo_aruco_state_space(self):
        # Servo (Position/Velocity/Load) + Servo XY + Target XY-Yaw + Goal
        state = []
        # remove the redundent target XY-Yaw + goal from the end
        servo_state_space = self.servo_state_space()[:-4]
        # remove the goal from the end
        aruco_state_space = self.aruco_state_space()[:-1]

        state += servo_state_space
        state += aruco_state_space

        state = self.add_goal(state)

        return state

    # TODO implement function
    def image_state_space(self):
        # Note should store the stacked frame somewhere...
        raise NotImplementedError("Requires implementation")

    @exception_handler("Failed to get state")
    def get_state(self):
        """
        Gets the current state of the environment based on the configured observation type (4 different options).

        Returns:
        A list representing the state of the environment.
        """
        if self.observation_type == OBSERVATION_TYPE.SERVO.value:
            return self.servo_state_space()
        elif self.observation_type == OBSERVATION_TYPE.ARUCO.value:
            return self.aruco_state_space()
        elif self.observation_type == OBSERVATION_TYPE.SERVO_ARUCO.value:
            return self.servo_aruco_state_space()
        elif self.observation_type == OBSERVATION_TYPE.IMAGE.value:
            return self.image_state_space()

        raise ValueError(f"Observation Type unknown: {self.observation_type}")

    def get_aruco_object_pose(self, blindable=False, detection_attempts=4):
        attempt = 0
        while not blindable or attempt < detection_attempts:
            attempt += 1
            msg = f"{attempt}/{detection_attempts}" if blindable else f"{attempt}"
            logging.debug(f"Attempting to detect aruco target: {self.object_marker_id}")

            self.last_frame = self.camera.get_frame()
            marker_poses = self.marker_detector.get_marker_poses(self.last_frame[:, :, [2, 1, 0]], self.camera.camera_matrix,
                                                                self.camera.camera_distortion, display=False)
            if self.object_marker_id in marker_poses:
                return marker_poses[self.object_marker_id]

            if attempt % 10 == 1:  # only log every 10th attempt, starting from second
                print(
                    f"Attempt {attempt} to detect object marker failed")
            if attempt > 25:
                self._return_to_home()
        return None

    def observed_object_state(self, marker_only=True):
        # A list representing the state of the observed object.
        object_state = self.get_aruco_object_pose(
            blindable=self.blindable, detection_attempts=5)
        if object_state is not None:
            state = []
            position = object_state["position"]
            orientation = object_state["orientation"]
            state.append(position[0])  # X
            state.append(position[1])  # Y
            state.append(orientation[2])  # Yaw

            if not marker_only:
                angle_offsets = [45, 135, 225, 315]
                state += self.get_object_ends_pose(orientation[2], angle_offsets, center_x=position[0],
                                                   center_y=position[1])
            return state
        return [0] * 11

    @exception_handler("Failed to get object states")
    def actual_object_state(self, yaw_only=True):
        yaw = self.target.get_yaw()
        if yaw_only:
            return yaw
        else:
            state = [yaw]
            angle_offsets = [0, 90, 180, 270]
            state += self.get_object_ends_pose(yaw, angle_offsets)
            return state

    def get_object_state(self, marker_only=True, yaw_only=True):
        if self.object_observation_mode == "observed":
            return self.observed_object_state(marker_only)
        elif self.object_observation_mode == "actual":
            return self.actual_object_state(yaw_only)
        else:
            raise ValueError("Object Observation Mode unknown")

    def get_object_ends_pose(self, center_yaw, angle_offsets, center_x=0, center_y=0):
        object_ends = [0] * 8
        ends_distance = 5.2

        for i in range(4):
            angle = center_yaw + angle_offsets[i]
            object_ends[i * 2] = center_x + np.sin(np.deg2rad(angle)) * ends_distance
            object_ends[i * 2 + 1] = center_y + np.cos(np.deg2rad(angle)) * ends_distance

        return object_ends

    def denormalize(self, action_norm):
        # return action in gripper range [-min, +max] for each servo
        action_gripper = [0 for _ in range(0, len(action_norm))]
        min_value_in = -1
        max_value_in = 1
        for i in range(0, self.gripper.num_motors):
            if self.action_type == "velocity":
                servo_min_value = self.gripper.velocity_min
                servo_max_value = self.gripper.velocity_max
            else:
                servo_min_value = self.gripper.min_values[i]
                servo_max_value = self.gripper.max_values[i]
            action_gripper[i] = int((action_norm[i] - min_value_in) * (servo_max_value - servo_min_value) / (
                        max_value_in - min_value_in) + servo_min_value)
        return action_gripper

    def normalize(self, action_gripper):
        # return action in algorithm range [-1, +1]
        max_range_value = 1
        min_range_value = -1
        action_norm = [0 for _ in range(0, len(action_gripper))]
        for i in range(0, self.gripper.num_motors):
            if self.action_type == "velocity":
                servo_min_value = self.gripper.velocity_min
                servo_max_value = self.gripper.velocity_max
            else:
                servo_min_value = self.gripper.min_values[i]
                servo_max_value = self.gripper.max_values[i]
            action_norm[i] = (action_gripper[i] - servo_min_value) * (max_range_value - min_range_value) / (
                        servo_max_value - servo_min_value) + min_range_value
        return action_norm

    def render(self):
        return self.camera.get_frame()

    def change_LED_colors(self, new_color):
        self.current_color = new_color
        self.gripper.change_LED_colours(new_color)
        self.elevator.LED_colour = new_color
        self.elevator.turn_on_LED()

    @abstractmethod
    def ep_final_distance(self):
        pass

    @abstractmethod
    def add_goal(self, state):
        pass

    @abstractmethod
    def choose_goal(self):
        pass

    @abstractmethod
    def reward_function(self, target, start_target_pose, final_target_pose):
        pass