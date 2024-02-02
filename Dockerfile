FROM nvidia/cuda:11.6.2-base-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Pacific/Auckland
RUN apt-get update
RUN apt-get -y install git python3 python3-pip
RUN apt-get update && apt-get -y install vim

RUN apt-get update && apt-get -y install ffmpeg libsm6 libxext6

# install cares_lib
RUN git clone https://github.com/UoA-CARES/cares_lib.git
RUN pip install -r /cares_lib/requirements.txt
RUN pip install /cares_lib

# install cares_reinforcement_learning
RUN git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
RUN pip install -r /cares_reinforcement_learning/requirements.txt
RUN pip install /cares_reinforcement_learning

# copy Gripper-Code
COPY . /Gripper-Code
RUN pip install -r /Gripper-Code/requirements.txt