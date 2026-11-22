FROM ros:humble-ros-base-jammy
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-gz ros-humble-message-filters python3-pip xvfb xauth libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /project
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY *.py viewer.html ./
COPY ros ./ros
COPY results/calibration.json results/settings.json ./results/
CMD ["bash", "ros/run.sh", "1000", "midas"]
