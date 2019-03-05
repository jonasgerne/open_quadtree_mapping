SET(PROJECT_NAME open_quadtree_mapping)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(${PROJECT_NAME})
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

option(WITH_ROS "Compile with ROS wrapper" True)

SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# find_package(CUDA REQUIRED)
# SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)

# Specify compute capability
# list(APPEND CUDA_NVCC_FLAGS -arch=sm_61)
# list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC;-std=c++11")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)
if(WITH_ROS)
  find_package(catkin REQUIRED COMPONENTS
    roscpp roslib cmake_modules cv_bridge image_transport pcl_ros sensor_msgs geometry_msgs)
  catkin_package(
      DEPENDS OpenCV Eigen Boost
      CATKIN_DEPENDS roscpp roslib image_transport pcl_ros
      INCLUDE_DIRS include
      # LIBRARIES efficient_map efficient_map_cuda
  )
endif()

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES")
else()
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# Set link libraries
list(APPEND LINK_LIBS
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)

include_directories(${catkin_INCLUDE_DIRS})
list(APPEND LINK_LIBS ${catkin_LIBRARIES})

add_library(open_quadtree_mapping_lib_depth STATIC
    src/check_cuda_device.cu
    src/seed_matrix.cu
    # src/bpextract.cu
    # src/depth_extract.cu
    # src/depth_fusion.cu
    # src/depth_upsample.cu
    # src/lsd_semidense.cu
    # src/quadtree.cu
    # src/seed_init.cu
    src/depthmap.cpp
)

add_library(open_quadtree_mapping_lib_inverse STATIC
        src/check_cuda_device.cu
        src/seed_matrix.cu
        # src/bpextract.cu
        # src/depth_extract.cu
        # src/depth_fusion.cu
        # src/depth_upsample.cu
        # src/lsd_semidense.cu
        # src/quadtree.cu
        # src/seed_init.cu
        src/depthmap.cpp
        )
target_compile_definitions(open_quadtree_mapping_lib_inverse PUBLIC "-DUSE_INVERSE_DEPTH")

if(WITH_ROS)
  add_executable(open_quadtree_mapping_node
      src/depthmap_node.cpp
      src/publisher.cpp
      src/main_ros.cpp
  )
  target_link_libraries(open_quadtree_mapping_node open_quadtree_mapping_lib_inverse
  ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
endif()

add_executable(open_quadtree_mapping_standalone_depth src/main_standalone.cpp)
target_link_libraries(open_quadtree_mapping_standalone_depth open_quadtree_mapping_lib_depth ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(open_quadtree_mapping_standalone_inverse src/main_standalone.cpp)
target_link_libraries(open_quadtree_mapping_standalone_inverse open_quadtree_mapping_lib_inverse ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_definitions(open_quadtree_mapping_standalone_inverse PRIVATE "-DUSE_INVERSE_DEPTH")

# Python module
include_directories( ${EIGEN3_INCLUDE_DIRS} )

#add_subdirectory(pybind11)
find_package(pybind11)
include_directories( ${pybind11_INCLUDE_DIRS} )
link_libraries( ${pybind11_LIBRARIES} )
pybind11_add_module(pyQuadtreeMapping src/python_wrapper.cpp)
TARGET_LINK_LIBRARIES(pyQuadtreeMapping PRIVATE open_quadtree_mapping_lib_depth ${OpenCV_LIBS} ${Boost_LIBRARIES})
