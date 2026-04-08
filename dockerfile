FROM ros:humble-ros-base

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-cv-bridge \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python packages
RUN pip3 install \
    "numpy==1.26.4" \
    "paddlepaddle==2.6.2" \
    "paddleocr==2.7.3"

# 3. Pre-cache PaddleOCR models to prevent runtime downloads
RUN python3 -c "\
import cv2, numpy as np; \
from paddleocr import PaddleOCR; \
ocr = PaddleOCR(use_angle_cls=True, lang='en'); \
dummy = np.zeros((100, 300, 3), dtype=np.uint8); \
ocr.ocr(dummy, cls=True)" || true

# 4. Setup ROS workspace
WORKDIR /ros_ws
RUN mkdir -p src

# 5. Copy packages
COPY ./ocr_interfaces /ros_ws/src/ocr_interfaces
COPY ./ocr_node /ros_ws/src/ocr_node

# 6. Build the workspace (interfaces first, then node)
RUN bash -c "\
    source /opt/ros/humble/setup.bash && \
    colcon build --packages-select ocr_interfaces && \
    source install/setup.bash && \
    colcon build --packages-select ocr_node"

# 7. Environment variables
ENV INPUT_TOPIC="/rgb"
ENV OUTPUT_TOPIC="/ocr_detection"
ENV CONFIDENCE_THRESHOLD="0.6"
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# 8. Setup sourcing
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros_ws/install/setup.bash" >> ~/.bashrc

# 9. Launch the node
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && ros2 run ocr_node ocr_node"]
