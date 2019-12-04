// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
#include <ros/ros.h>
#include <quadmap/check_cuda_device.cuh>
#include <quadmap/depthmap_node.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PoseStamped.h>

#include <image_transport/image_transport.h>

typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, geometry_msgs::PoseStamped> exact_policy;

int main(int argc, char **argv) {
    if (!quadmap::checkCudaDevice(argc, argv))
        return EXIT_FAILURE;

    ros::init(argc, argv, "hybrid_mapping");
    ros::NodeHandle nh("~");
    image_transport::ImageTransport it_(nh);
    quadmap::DepthmapNode dm_node(nh);
    image_transport::Subscriber sub_;
    message_filters::Synchronizer<exact_policy> sync_(exact_policy(1000));
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> pose_sub_;
    bool use_tf_transforms_;

    if (!dm_node.init()) {
        ROS_ERROR("could not initialize DepthmapNode. Shutting down node...");
        return EXIT_FAILURE;
    }
    ROS_INFO("Init ok.");

    nh.param("use_tf_transforms", use_tf_transforms_, false);
    if (!use_tf_transforms_) {
        image_sub_.subscribe(nh, "image", 1000);
        pose_sub_.subscribe(nh, "posestamped", 1000);
        sync_.connectInput(image_sub_, pose_sub_);
        sync_.registerCallback(boost::bind(&quadmap::DepthmapNode::Msg_Callback, &dm_node, _1, _2));
    } else {
        std::string image_topic = nh.resolveName("image");
        std::string cam_info_topic = ros::names::parentNamespace(image_topic) + "/camera_info";
        boost::shared_ptr<const sensor_msgs::CameraInfo> cam_info_ptr = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(
                cam_info_topic, nh, ros::Duration(3));
        dm_node.setFrameName(cam_info_ptr->header.frame_id);
        sub_ = it_.subscribe(image_topic, 10, &quadmap::DepthmapNode::imageCb, &dm_node);
        ROS_INFO("Topic: %s, NumPub: %d, Transport: %s", sub_.getTopic().c_str(), sub_.getNumPublishers(),
                 sub_.getTransport().c_str());
    }

    while (ros::ok()) {
        ros::spinOnce();
    }
    return EXIT_SUCCESS;
}
