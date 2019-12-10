FROM ros_cuda

RUN git clone https://github.com/HKUST-Aerial-Robotics/open_quadtree_mapping.git
WORKDIR /home/camp/catkin_ws

RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash ;\
                  catkin_make ;"

RUN /bin/bash -c 'echo "source /opt/ros/kinetic/setup.bash;\
                  source /home/camp/catkin_ws/devel/setup.sh" > /home/camp/.bashrc'

