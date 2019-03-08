FROM cuda-10-cudnn7

# Install ROS
RUN /bin/bash -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
RUN apt-get update
RUN apt-get install -y ros-kinetic-desktop
RUN apt-get install -y ros-kinetic-pcl-ros

RUN rosdep init

RUN apt-get install -y git python-rosinstall python-rosinstall-generator python-wstool build-essential

# Replacing shell with bash for later docker build commands
RUN mv /bin/sh /bin/sh-old && \
ln -s /bin/bash /bin/sh

# User
RUN useradd -ms /bin/bash camp
USER camp

RUN rosdep update

RUN mkdir -p /home/camp/catkin_ws/src
WORKDIR /home/camp/catkin_ws/src

RUN git clone https://github.com/HKUST-Aerial-Robotics/open_quadtree_mapping.git
WORKDIR /home/camp/catkin_ws
#RUN source /opt/ros/kinetic/setup.bash
#RUN catkin_init_workspace
#RUN catkin_make

RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash ;\
                  catkin_make ;"

RUN /bin/bash -c 'echo "source /opt/ros/kinetic/setup.bash;\
                  source /home/camp/catkin_ws/devel/setup.sh" > /home/camp/.bashrc'

