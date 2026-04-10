FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevents ANY apt interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y \
      ros-humble-ros-base \
      python3-colcon-common-extensions \
      python3-pip \
      ros-humble-cv-bridge \
      ros-humble-vision-msgs \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    "numpy==1.26.4" \
    "paddleocr==2.7.3" \
    paddlepaddle-gpu==2.6.1.post117 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

RUN python3 -c "\
from paddleocr import PaddleOCR; \
PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)"

WORKDIR /ros_ws
RUN mkdir -p src
COPY ./ocr_node /ros_ws/src/ocr_node

RUN bash -c "\
    source /opt/ros/humble/setup.bash && \
    colcon build"

ENV INPUT_TOPIC="/rgb"
ENV CONFIDENCE_THRESHOLD="0.6"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros_ws/install/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && ros2 run ocr_node ocr_node"]
