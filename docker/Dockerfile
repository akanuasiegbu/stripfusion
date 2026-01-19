From pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update && apt-get -y install wget

RUN python -m  pip install --upgrade pip

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install PyYAML==5.3.1 
RUN pip install protobuf==3.20.0
RUN pip install gpustat


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 nano  -y
# RUN apt-get update && apt-get install -y apt-utils
RUN apt-get -y install git
RUN pip install ipykernel
RUN pip install einops
RUN pip install ptflops==0.7.2.2 
RUN pip install scikit-learn==1.3.2

RUN apt-get install -y python3-tk
# ipykernel and ipython needed for ipynb to run in devcontainer
RUN apt-get -y install tmux
RUN apt-get -y install htop
RUN pip install timm

ARG USERNAME=njotwani
ARG USER_UID=995200167
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

