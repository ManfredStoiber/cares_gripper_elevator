<h1 align="center">CARES Gripper Package</h1>

<div align="center">

[![Python 3.11.4](https://img.shields.io/badge/python-3.11.4-blue.svg)](https://www.python.org/downloads/release/python-3114/)
[![Pytorch 1.13.1](https://img.shields.io/badge/pytorch-1.13.1-blue)](https://pytorch.org/)

</div>

<div align="center">
<h3>
<a href="https://www.youtube.com/watch?v=0kii1EJjOzw&feature=youtu.be" target="_blank">Video Demo</a>
</h3>
</div>

<div align="center">
This repository contains the code used to control and test the grippers (Two-finger and Three-Finger) currently being designed and used in the <a href="https://cares.blogs.auckland.ac.nz/">CARES lab</a> at the <a href="https://www.auckland.ac.nz">The University of Auckland</a>. 
While being written for this specific system, it also intends to be applicable to many dynamixel servo systems with minor changes to the code.

<br/>
<br/>
See the gripper in action, learning to rotate the valve by 90 degrees:
<br/>
<br/>

| Exploration Phase                                                                      | During Training                                                                     | Final Policy                                                                      |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| <img src="readme_wiki_media/exploration_phase_final.gif" alt="explore" height="500px"> | <img src="readme_wiki_media/during_training_final.gif" alt="during" height="500px"> | <img src="readme_wiki_media/trained_policy_final.gif" alt="final" height="500px"> |

</div>

## Contents

- [Contents](#contents)
- [📋 Requirements](#-requirements)
- [👩‍🏫 Getting Started](#-getting-started)
- [📖 Usage](#-usage)
- [⚙️ Hardware Setup](#️-hardware-setup)
  - [Magnetic Encoder Setup](#magnetic-encoder-setup)
  - [BOM](#bom)
  - [STL files](#stl-files)
- [🗃️ Results](#️-results)
- [📈 Benchmarking Graphs](#-benchmarking-graphs)
  - [Instructions](#instructions)
- [📦 Package Structure](#-package-structure)
  - [Folders](#folders)
  - [Files](#files)

## 📋 Requirements

The repository was tested using Python 3.11.4 on a machine running Ubuntu 22.04.2 LTS with Intel Core i9-10900 CPU and NVIDIA GeForce RTX 3080 GPU. It is recommended to use a Linux machine. The repository relies on [Pytorch](https://pytorch.org/). While the use of [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-zone) is supported, it is optional. Instructions for enabling CUDA in Pytorch can be found [here](https://pytorch.org/get-started/locally/).

A comprehensive list of dependencies is available in `requirements.txt`. Ensure that the hardware components for the gripper are turned on and connected to the machine.

## 👩‍🏫 Getting Started

1. Clone the repository using `git clone`.

2. Run `pip3 install -r requirements.txt` in the **root directory** of the package.

3. Install the two CARES libraries as instructed in [CARES Lib](https://github.com/UoA-CARES/cares_lib) and [CARES Reinforcement Learning Package](https://github.com/UoA-CARES/cares_reinforcement_learning).

4. Create a folder named `your_folder_name` to store config files. To get started, copy and paste the files in `scripts/config_examples` into the folder. For a guide on changing configs, see the [wiki](https://github.com/UoA-CARES/Gripper-Code/wiki/Configuration-Files).

## 📖 Usage

Consult the repository's [wiki](https://github.com/UoA-CARES/Gripper-Code/wiki) for a guide on how to use the package.

## ⚙️ Hardware Setup

The current setup uses Dynamixel XL-320 servo motors (4 for Two-Finger and 9 for Three-Finger Gripper), which are being controlled using a [U2D2](https://emanual.robotis.com/docs/en/parts/interface/u2d2/).
Below is an example of the wiring setup for the three-fingered gripper. This is easily changed for other configurations, with a maximum of 4 daisy chains coming from the U2D2.

<img src="https://github.com/UoA-CARES/Gripper-Code/assets/105029122/994e451f-8459-42e2-9aa7-c27b7d10af29" width="400" />

### Magnetic Encoder Setup

An AS5600 magnetic encoder can be used to get the object angle during training. 3D printed object valve suitable for using this encoder can be found in the STL files folder below.

To set this up

1. Connect the encoder with an arduino board. VCC - 3.3V; GND - GND; DIR - GND; SCL - SCL; SDA - SDA; (see wiring digram below)
2. Upload magnetic_encoder_object.ino onto the Arduino.
3. Check device name and modify the object_config file accordingly

   <img src="https://github.com/UoA-CARES/Gripper-Code/assets/105029122/305bc589-e68e-4433-9fbd-919544614493" alt="wiring diagram for connecting an as5600 magnetic encoder to an Arduino Mega" width="400" />

### BOM

A list of items required to build the grippers can be found in [Grippers BOM](https://docs.google.com/spreadsheets/d/1GFGDXZwodSCUbbnDEK6e9giJs_8Xy-eVyAdYuDRv4Qk/edit#gid=1627805202).

### STL files

3D printed parts for both grippers can be found in [Two-Finger STL](https://drive.google.com/drive/folders/1AuBA8254ImEZFrz9au1Tdciz5qx39S2c?usp=share_link) and [Three-Finger STL](https://drive.google.com/drive/folders/1AuBA8254ImEZFrz9au1Tdciz5qx39S2c?usp=share_link).

![Picture of a CAD assembly that shows a rig that is holding a three-fingered gripper with the fingers hanging down](https://user-images.githubusercontent.com/105029122/205157459-ef70f9fb-dcea-464a-af8a-14d66047497a.png)

## 🗃️ Results

You can specify the folder to save results in using the `local_results_path ` argument; otherwise, it defaults to `{home_path}/gripper_training`. The folder containing the results is named according to the following convention:

```
{date}_{robot_id}_{environment_type}_{observation_type}_{seed}_{algorithm}
```

The structure is shown below:

```
result_folder_name/
├─ configs/
├─ data/
│  ├─ {result_folder_name}_evaluation
│  ├─ distance.txt
│  ├─ reward.txt
│  ├─ steps_per_episode.txt
│  ├─ success_list.txt
│  ├─ time.txt
│  ├─ rolling_reward_average.txt
│  ├─ rolling_steps_per_episode_average.txt
│  ├─ rolling_success_average.txt
├─ models/
├─ ...
```

Descriptions of each directory and file are as follows:

`configs/`: Contains source config files used during the model training.

`data/`: Stores raw data produced during the training process.

`models/`: Stores files for the best models obtained based on reward, primarily used for evaluation purposes.

`{result_folder_name}_evaluation`: Represents a CSV file containing data from the evaluation episodes during the training phase.

`distance.txt`: Provides the recorded distance between the current valve angle and the goal angle at the end of each episode.

`reward.txt`: Captures the raw reward value at the conclusion of each episode.

`steps_per_episode.txt`: Documents the number of steps taken per episode.

`success_list.txt`: Contains a binary indicator; 1 denotes successful completion of the goal, while 0 indicates failure for each episode.

`time.txt`: Records the time taken to complete each episode.

`rolling_reward_average.txt`, `rolling_steps_per_episode_average.txt`, `rolling_success_average`: These files contain the rolling averages over a fixed window size for the recorded rewards, steps per episode, and success rates, respectively.

## 📈 Benchmarking Graphs

The `graph_results.py` file contains utility functions for graphing benchmarking results. You can use it to visualise data from different files based on specific plot types. Here are the supported values for the `plot_type` parameter:

- `reward`: Plots the data from the `reward.txt` file over a moving window of size 20.
- `success_rate`: Plots the data from the `success_list.txt` file over a moving window of size 100.
- `training_evaluation`: Plots data from the evaluation episode, occurring every 10 training episodes, over a moving window of size 20.

### Instructions

1. Organise your data into a folder structure similar to the following:

```
folder_path/
├─ training/
│  ├─ 1_90/
│  │  ├─ 08_07_16_58_RobotId2_EnvType0_ObsTypeservo_Seed1_SAC/
│  │  ├─ 08_07_16_58_RobotId4_EnvType0_ObsTypeservo_Seed10_DDPG/
│  │  ├─ 09_11_10_09_RobotId2_EnvType0_ObsTypeservo_Seed10_TD3/
│  ├─ 2_90,180,270/
│  │  ├─ ...
│  ├─ 3_30-330/
│  │  ├─ ...
├─ eval/
│  ├─ 1_90/
│  │  ├─ ...
│  ├─ 2_90,180,270/
│  │  ├─ ...
│  ├─ 3_30-330/
│  │  ├─ ...
```

2. From the `scripts` directory, run the command `python3 graph_results.py --path folder_path --plot_type plot_type`.

3. The resulting graph will be saved in the `graphs` folder located in the root directory.

## 📦 Package Structure

```
cares_gripper_package/scripts/
├─ config_examples/
│  ├─ env_xDOF_config_IDx.json
│  ├─ gripper_9DOF_config_ID2.json (3 Fingered Gripper)
│  ├─ gripper_xDOF_config_IDx.json
│  ├─ learning_config_IDx.json
│  ├─ object_config_IDx.json
│  ├─ camera_distortion.txt
│  ├─ camera_matrix.txt
├─ environments/
│  ├─ Environment.py
│  ├─ RotationEnvironment.py
│  ├─ TranslationEnvironment.py
│  ├─ ...
├─ networks/
│  ├─ DDPG/
│  ├─ ├─ Actor.py
│  ├─ ├─ Critic.py
│  ├─ SAC/
│  ├─ ├─ ...
│  ├─ TD3/
│  ├─ ├─ ...
├─ magnetic_encoder_object
│  ├─ magnetic_encoder_object.ino
├─ tools
│  ├─ error_handlers.py
│  ├─ utils.py
├─ configurations.py
├─ evaluation_loop.py
├─ gripper_example.py
├─ GripperTrainer.py
├─ Objects.py
├─ training_loop.py
├─ graph_results.py
```

### Folders

`config_examples/`: Various configuration file examples for the environment, gripper, training and camera. Instructions for these configs can be found in [wiki]().

`environments/`: Currently for rotation and translation tasks. Can extend environment class for different tasks by changing choose goal and reward function.

`networks/`: Can contain your own neural networks that can be used with each algorithm. Currently, we support the DDPG, SAC, and TD3 RL algorithms.

`magnetic_encoder_object/`: Currently only contains arduino code for the magnetic encoder.

`tools/`: Includes helper functions for I/O, plotting, and Slack integration in `utils.py`. Functions to handle various gripper errors and Slack monitoring can be found in `error_handlers.py`.

### Files

`configurations.py`: Pydantic class models for the learning, environment, and object configurations.

`training_loop.py`: Parses configuration files and starts training for the selected algorithm and configs.

`evaluation_loop.py`: Parses configuration files and the trained model for the **final policy** and evaluates it.

`graph_results.py`: Various graphing functions for benchmarking the performance of RL algorithms side by side

`gripper_example.py`: Example of moving the Gripper

`GripperTrainer.py`: Class initialising necessary components for training an RL algorithm and controlling the gripper. Logs information, and performs training and evaluation loops with Slack messaging for updates during the process.

`Objects.py`: Class definitions for Servo, Aruco, and Magnet objects.
