import cv2
from environments.Environment import Environment
from environments.RotationEnvironment import fixed_goals
from environments.RotationEnvironment import RotationEnvironment
import logging
import numpy as np

from pathlib import Path

file_path = Path(__file__).parent.resolve()

from configurations import EnvironmentConfig, ObjectConfig
from cares_lib.dynamixel.gripper_configuration import GripperConfig


class TranslationEnvironment(Environment):
    def __init__(self, env_config: EnvironmentConfig, gripper_config: GripperConfig, object_config: ObjectConfig):
        super().__init__(env_config, gripper_config, object_config)
        self.goal_state = self.get_object_state()

    # overriding method
    def choose_goal(self):
        object_state = self.get_object_state()

        position = object_state[0:2]
        position[0] = np.random.randint(-50, 50)
        position[1] = np.random.randint(-70, -40)

        return position

        if self.object_observation_mode == "observed":
            yaw = object_state[-1]

        yaw_goal = fixed_goals(yaw, self.noise_tolerance)

        goal = position + yaw_goal

        return goal

    def reward_function(self, target_goal, goal_before, goal_after):
        if goal_before is None:
            logging.debug("Start Marker Pose is None")
            return 0, True

        if goal_after is None:
            logging.debug("Final Marker Pose is None")
            return 0, True
        done = False

        target_goal_position = np.array(target_goal[0:2])
        target_goal_yaw = np.array(target_goal[-1])

        goal_position_after_array = np.array(goal_after[0:2])
        goal_position_difference = np.linalg.norm(target_goal_position - goal_position_after_array)

        reward_position = -goal_position_difference

        if self.object_observation_mode == "observed":
            yaw_after_rounded = round(goal_after[-1])
        elif self.object_observation_mode == "actual":
            yaw_after_rounded = round(goal_after)

        goal_yaw_difference = RotationEnvironment.rotation_min_difference(target_goal_yaw, yaw_after_rounded)

        reward_yaw = -goal_yaw_difference/2


        # The following step might improve the performance.

        # goal_before_array = goal_before[0:2]
        # delta_changes   = np.linalg.norm(target_goal - goal_before_array) - np.linalg.norm(target_goal - goal_after_array)
        # if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
        #     reward = -10
        # else:
        #     reward = -goal_position_difference
        #     #reward = delta_changes / (np.abs(yaw_before - target_goal))
        #     #reward = reward if reward > 0 else 0

        # For Translation. noise_tolerance is 15, it would affect the performance to some extent.
        if goal_position_difference <= self.noise_tolerance and goal_yaw_difference <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            done = True
            reward = 500
        else:
            reward = (reward_position + reward_yaw) /2

        logging.info(f"Reward: {reward}, Goal after: {goal_position_after_array}")

        return reward, done

    def ep_final_distance(self):
        return np.linalg.norm(np.array(self.goal_state) - np.array(self.get_object_state()[0:2]))

    def add_goal(self, state):
        state.append(self.goal_state[0])
        state.append(self.goal_state[1])
        return state

    def render(self):
        if self.last_frame is None:
            self.last_frame = self.camera.get_frame()

        frame = self.last_frame.copy()

        while True:

            marker_poses = self.marker_detector.get_marker_poses(frame, self.camera.camera_matrix,
                                                                 self.camera.camera_distortion, display=False)

            if self.object_marker_id in marker_poses:
                break

            frame = self.camera.get_frame()

        imgpoints, _ = cv2.projectPoints(np.array([[self.goal_state[0], self.goal_state[1], marker_poses[self.object_marker_id]['position'][2]]], dtype=float), np.array([.0, .0, .0]), np.array([.0, .0, .0]),
                                             self.camera.camera_matrix, self.camera.camera_distortion)

        cv2.circle(frame, np.array(imgpoints[0][0], dtype=int), 1, (0, 0, 255), thickness=20)

        return frame

    def env_render(self, done=False, step=1, episode=1, mode="Exploration"):
        image = self.camera.get_frame()
        color = (0, 255, 0)
        if done:
            color = (0, 0, 255)

        target = (int(self.goal_pixel[0]), int(self.goal_pixel[1]))
        text_in_target = (
            int(self.goal_pixel[0]) - 15, int(self.goal_pixel[1]) + 3)
        cv2.circle(image, target, 18, color, -1)
        cv2.putText(image, 'Target', text_in_target, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Episode : {str(episode)}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(image, f'Steps : {str(step)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Success Counter : {str(self.counter_success)}', (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Stage : {mode}', (900, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("State Image", image)
        cv2.waitKey(10)

