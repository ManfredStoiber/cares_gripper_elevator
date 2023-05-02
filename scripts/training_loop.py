import logging
logging.basicConfig(level=logging.INFO)

import os
import pydantic
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

import torch
import random
import numpy as np

from queue import Queue, Empty

from threading import Thread

from environments.RotationEnvironment import RotationEnvironment
from environments.TranslationEnvironment import TranslationEnvironment
from configurations import LearningConfig, EnvironmentConfig, GripperConfig

from environments.Environment import EnvironmentError
from Gripper import GripperError
import error_handlers as erh

from cares_reinforcement_learning.algorithm import TD3
from networks import Actor
from networks import Critic
from cares_reinforcement_learning.util import MemoryBuffer
from cares_lib.slack_bot.SlackBot import SlackBot

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logging.info("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    logging.info("Working with CPU")
 
with open('slack_token.txt') as file: 
    slack_token = file.read()
slack_bot = SlackBot(slack_token=slack_token)

# TODO consider - https://github.com/rustedpy/result
class ErrorWrapper():
    def __init__(self, value, error=None):
        self.value = value
        self.error = error

    def unwrap(self):
        if self.error is not None:
            raise self.error
        return self.value

class GripperTrainer():
    def __init__(self, env_config, gripper_config, learning_config, file_path) -> None:
        self.learning_config = learning_config #TODO split this out into the actual values to reduce the self.learning_config stuff below

        if env_config.env_type == 0:
            self.environment = RotationEnvironment(env_config, gripper_config)
        elif env_config.env_type == 1:
            self.environment = TranslationEnvironment(env_config, gripper_config)

        logging.info("Resetting Environment")
        state = self.environment.reset()#will just crash right away if there is an issue but that is fine
        
        logging.info(f"State: {state}")
        slack_bot.post_message("#bot_terminal", f"#{self.environment.gripper.gripper_id}: Reset Terminal. \nState: {state}")

        observation_size = len(state)# This wont work for multi-dimension arrays
        action_num       = gripper_config.num_motors
        message = f"Observation Space: {observation_size} Action Space: {action_num}"
        logging.info(message)
        slack_bot.post_message("#bot_terminal", f"#{self.environment.gripper.gripper_id}: {message}")

        logging.info("Setting up Network")
        actor  = Actor(observation_size, action_num, learning_config.actor_lr)
        critic = Critic(observation_size, action_num, learning_config.critic_lr)

        logging.info("Setting up Memory")
        self.memory = MemoryBuffer(learning_config.buffer_capacity)

        logging.info("Setting RL Algorithm")
        self.agent = TD3(
            actor_network=actor,
            critic_network=critic,
            gamma=learning_config.gamma,
            tau=learning_config.tau,
            action_num=action_num,
            device=DEVICE,
        )

        self.file_path = file_path

        logging.info("Strating Environment Thread")
        self.action_queue = Queue(maxsize=1)
        self.state_queue  = Queue(maxsize=1)

        self.environment_thread = Thread(target = self.environment_spin)
        self.environment_thread.start()

    # TODO tidy up the error handling within these functions
    def step_envrionment(self, action):
        if not self.environment_thread.is_alive():
            self.environment_thread.start()

        try:
            action_command = [1, action]
            self.action_queue.put(action_command)
            return self.state_queue.get(timeout=5).unwrap()
        except (EnvironmentError , GripperError) as error:
            error_message = f"Failed to step with message: {error}"
            logging.error(error_message)
            if erh.handle_gripper_error_home(self.environment, error_message, slack_bot, self.file_path):
                return [], 0, True, True
            else:
                self.environment.gripper.close() # can do more if we want to save states and all
                exit()
        except Empty as error:
            logging.error(f"Timed out waiting for state after step - environment thread is Alive: {self.environment_thread.is_alive()}")
            if not self.environment_thread.is_alive():
                self.environment_thread.start()
            self.reset_envrionment()
            return [], 0, True, True

    # TODO tidy up the error handling within these functions
    def reset_envrionment(self):
        if not self.environment_thread.is_alive():
            self.environment_thread.start()

        reset_command = [0, None]
        self.action_queue.put(reset_command)
        
        try:
            return self.state_queue.get(timeout=5).unwrap()
        except (EnvironmentError , GripperError) as error:
            error_message = f"Failed to reset with message: {error}"
            logging.error(error_message)
            if erh.handle_gripper_error_home(self.environment, error_message, slack_bot, self.file_path):
                return self.reset_envrionment(self.environment, self.file_path)  # might keep looping if it keep having issues
            else:
                self.agent.save_models(self.file_path)#TODO this needs to be updated to take a file_path
                exit()
        except Empty as error:
            logging.error(f"Timed out waiting for state after reset - environment thread is Alive: {self.environment_thread.is_alive()}")
            if not self.environment_thread.is_alive():
                self.environment_thread.start()
            self.reset_envrionment()

    def environment_spin(self):
        while True:
            try:
                try:
                    command, action = self.action_queue.get(False)
                    if command == 0:
                        value = self.environment.reset()
                    elif command == 1:
                        value = self.environment.step(action)
                    self.state_queue.put(ErrorWrapper(value))
                except Empty as error:
                    pass
            
                self.environment.gripper_step()
            except (EnvironmentError , GripperError) as error:
                logging.error(f"Environment thread error: {error}")        
                self.state_queue.put(ErrorWrapper(None, error))
                return

    def evaluation(self):
        logging.info("Starting Training Loop")
        self.agent.load_models(filename=self.file_path)

        episode_timesteps = 0
        episode_reward    = 0
        episode_num       = 0

        state = self.environment_reset() 

        max_steps_evaluation        = 100
        episode_horizont_evaluation = 20

        for total_step_counter in range(max_steps_evaluation):
            episode_timesteps += 1

            action = self.agent.select_action_from_policy(state, evaluation=True)  # algorithm range [-1, 1]
            action_env = self.environment.denormalize(action)  # gripper range

            next_state, reward, done, truncated = self.step_envrionment(action_env)
            if not truncated:
                logging.info(f"Reward of this step:{reward}")
                state = next_state
                episode_reward += reward

            if done or truncated or episode_timesteps >= episode_horizont_evaluation:
                logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

                # Reset environment
                state =  self.reset_envrionment()

                episode_reward    = 0
                episode_timesteps = 0
                episode_num += 1

    def train(self):
        logging.info("Starting Training Loop")

        episode_timesteps = 0
        episode_reward    = 0
        episode_num       = 0
        historical_reward = {"step": [], "episode_reward": []}
        best_episode_reward = -np.inf

        # TODO extract these parameters
        min_noise    = 0.01
        noise_decay  = 0.9999
        noise_scale  = 0.10

        state = self.reset_envrionment()

        for total_step_counter in range(int(self.learning_config.max_steps_training)):
            episode_timesteps += 1

            if total_step_counter < self.learning_config.max_steps_exploration:
                message = f"Running Exploration Steps {total_step_counter}/{self.learning_config.max_steps_exploration}"
                logging.info(message)
                if total_step_counter%50 == 0:
                    slack_bot.post_message("#bot_terminal", f"#{self.environment.gripper.gripper_id}: {message}")
                
                action_env = self.environment.sample_action()
                action = self.environment.normalize(action_env) # algorithm range [-1, 1]
            else:
                noise_scale *= noise_decay
                noise_scale = max(min_noise, noise_scale)
                logging.info(f"Noise Scale:{noise_scale}")

                message = f"Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n"
                logging.info(message)

                action = self.agent.select_action_from_policy(state, noise_scale=noise_scale)  # algorithm range [-1, 1]
                action_env = self.environment.denormalize(action)  # gripper range
            
            next_state, reward, done, truncated = self.step_envrionment(action_env)

            if not truncated:
                logging.info(f"Reward of this step:{reward}")

                self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

                state = next_state

                episode_reward += reward

                if self.environment.action_type == "position": # if position based, train every step
                    if total_step_counter >= self.learning_config.max_steps_exploration:
                        for _ in range(self.learning_config.G):
                            experiences = self.memory.sample(self.learning_config.batch_size)
                            self.agent.train_policy(experiences)
                        
            if done or truncated or episode_timesteps >= self.learning_config.episode_horizont:
                message = f"#{self.environment.gripper.gripper_id} - Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}"
                logging.info(message)
                slack_bot.post_message("#bot_terminal", message)

                historical_reward["step"].append(total_step_counter)
                historical_reward["episode_reward"].append(episode_reward)

                if self.environment.action_type == "velocity": # if velocity based, train every episode
                    if total_step_counter >= self.learning_config.max_steps_exploration:
                        for _ in range(self.learning_config.G):
                            experiences = self.memory.sample(self.learning_config.batch_size)
                            self.agent.train_policy(experiences)

                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    self.agent.save_models(self.file_path)#TODO this needs to be updated to take a file_path

                state = self.reset_envrionment()

                episode_reward    = 0
                episode_timesteps = 0
                episode_num      += 1

                if episode_num % self.learning_config.plot_freq == 0:
                    plot_reward_curve(historical_reward, self.file_path)

        plot_reward_curve(historical_reward, self.file_path, "rewards")
        self.agent.save_models(self.file_path)#TODO this needs to be updated to take a file_path
        self.environment.gripper.close()

# todo move this function to better place
def create_directories(local_results_path, folder_name):
    if not os.path.exists(local_results_path):
        os.makedirs(local_results_path)

    file_path = f"{local_results_path}/{folder_name}"
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists("servo_errors"): #servo error still here because it's used by servo.py which shouldn't know the local storage
        os.makedirs("servo_errors")

# todo move this function to better place
def plot_reward_curve(data_reward, file_path, file_name):
    data = pd.DataFrame.from_dict(data_reward)
    data.to_csv(f"{file_path}/{file_name}", index=False)
    data.plot(x='step', y='episode_reward', title=file_name)
    plt.savefig(f"{file_path}/{file_name}")
    plt.close()

def store_configs(file_path, env_config, gripper_config, learning_config):
    with open(f"{file_path}/configs.txt", "w") as f:
        f.write(f"Environment Config:\n{env_config.json()}\n")
        f.write(f"Gripper Config:\n{gripper_config.json()}\n")
        f.write(f"Learning Config:\n{learning_config.json()}\n")
        with open(Path(env_config.camera_matrix)) as cm:
            f.write(f"\nCamera Matrix:\n{cm.read()}\n")
        with open(Path(env_config.camera_distortion)) as cd:
            f.write(f"Camera Distortion:\n{cd.read()}\n")
        
def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--learning_config", type=str)
    parser.add_argument("--env_config",      type=str)
    parser.add_argument("--gripper_config",  type=str)

    home_path = os.path.expanduser('~')
    parser.add_argument("--local_results_path",  type=str, default=f"{home_path}/gripper_training")
    return parser.parse_args()

def main():

    args = parse_args()
    env_config      = pydantic.parse_file_as(path=args.env_config,      type_=EnvironmentConfig)
    gripper_config  = pydantic.parse_file_as(path=args.gripper_config,  type_=GripperConfig)
    learning_config = pydantic.parse_file_as(path=args.learning_config, type_=LearningConfig)
    local_results_path = args.local_results_path

    logging.info("Setting up Seeds")
    torch.manual_seed(learning_config.seed)
    np.random.seed(learning_config.seed)
    random.seed(learning_config.seed)

    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    #TODO add the agent/algorithm type to learning_config
    file_path  = f"{date_time_str}_"
    file_path += f"RobotId{gripper_config.gripper_id}_EnvType{env_config.env_type}_ObsType{env_config.object_type}_Seed{learning_config.seed}_TD3"

    create_directories(local_results_path, file_path)
    store_configs(file_path, env_config, gripper_config, learning_config)

    gripper_trainer = GripperTrainer(env_config, gripper_config, learning_config, file_path)
    gripper_trainer.train()

    #gripper_trainner.evaluation()

if __name__ == '__main__':
    main()