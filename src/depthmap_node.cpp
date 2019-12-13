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
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <quadmap/depthmap_node.h>

#include <quadmap/se3.cuh>

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <string>
#include <future>


using namespace std;

quadmap::DepthmapNode::DepthmapNode(ros::NodeHandle &nh)
        : nh_(nh), num_msgs_(0)
//  , it_(nh_)
{}

bool quadmap::DepthmapNode::init() {
    int cam_width;
    int cam_height;
    float cam_fx;
    float cam_fy;
    float cam_cx;
    float cam_cy;
    double downsample_factor;
    int semi2dense_ratio;
    std::string tf_goal_frame_;
    nh_.getParam("cam_width", cam_width);
    nh_.getParam("cam_height", cam_height);
    nh_.getParam("cam_fx", cam_fx);
    nh_.getParam("cam_fy", cam_fy);
    nh_.getParam("cam_cx", cam_cx);
    nh_.getParam("cam_cy", cam_cy);

    nh_.getParam("downsample_factor", downsample_factor);
    nh_.getParam("semi2dense_ratio", semi2dense_ratio);

    printf("read : width %d height %d\n", cam_width, cam_height);

    float k1, k2, r1, r2;
    k1 = k2 = r1 = r2 = 0.0;
    if (nh_.hasParam("cam_k1") &&
        nh_.hasParam("cam_k2") &&
        nh_.hasParam("cam_r1") &&
        nh_.hasParam("cam_r2")) {
        nh_.getParam("cam_k1", k1);
        nh_.getParam("cam_k2", k2);
        nh_.getParam("cam_r1", r1);
        nh_.getParam("cam_r2", r2);
    }

    // initial the remap mat, it is used for undistort and also resive the image
    cv::Mat input_K = (cv::Mat_<float>(3, 3) << cam_fx, 0.0f, cam_cx, 0.0f, cam_fy, cam_cy, 0.0f, 0.0f, 1.0f);
    cv::Mat input_D = (cv::Mat_<float>(1, 4) << k1, k2, r1, r2);

    float resize_fx, resize_fy, resize_cx, resize_cy;
    resize_fx = cam_fx * downsample_factor;
    resize_fy = cam_fy * downsample_factor;
    resize_cx = cam_cx * downsample_factor;
    resize_cy = cam_cy * downsample_factor;
    cv::Mat resize_K = (cv::Mat_<float>(3, 3)
            << resize_fx, 0.0f, resize_cx, 0.0f, resize_fy, resize_cy, 0.0f, 0.0f, 1.0f);
    resize_K.at<float>(2, 2) = 1.0f;
    int resize_width = cam_width * downsample_factor;
    int resize_height = cam_height * downsample_factor;

    cv::Mat undist_map1, undist_map2;
    cv::initUndistortRectifyMap(
            input_K,
            input_D,
            cv::Mat_<double>::eye(3, 3),
            resize_K,
            cv::Size(resize_width, resize_height),
            CV_32FC1,
            undist_map1, undist_map2);

    depthmap_ = std::make_shared<quadmap::Depthmap>(resize_width, resize_height, 4, resize_fx, resize_cx, resize_fy,
                                                    resize_cy, undist_map1, undist_map2, semi2dense_ratio,
                                                    false, false, false, false, false,
                                                    true, 0.003f, 0.01f, false, 1.0, 50.0,
                                                    0.95f, 0.03f, 0.95f,
                                                    0.03f, 0.6, 0.45, 1.0f,
                                                    1.0f, 0.0f);

    bool pub_pointcloud = false;
    nh_.getParam("publish_pointcloud", pub_pointcloud);
    publisher_.reset(new quadmap::Publisher(nh_, depthmap_));

    return true;
}


void quadmap::DepthmapNode::Msg_Callback(
        const sensor_msgs::ImageConstPtr &image_input,
        const geometry_msgs::PoseStampedConstPtr &pose_input) {
    printf("\n\n\n");
    num_msgs_ += 1;
    current_msg_time = image_input->header.stamp;
    if (!depthmap_) {
        ROS_ERROR("depthmap not initialized. Call the DepthmapNode::init() method");
        return;
    }
    cv::Mat img_8uC1;
    try {
        cv_bridge::CvImageConstPtr cv_img_ptr =
                cv_bridge::toCvShare(image_input, sensor_msgs::image_encodings::MONO8);
        img_8uC1 = cv_img_ptr->image;
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    quadmap::SE3<float> T_world_curr(
            pose_input->pose.orientation.w,
            pose_input->pose.orientation.x,
            pose_input->pose.orientation.y,
            pose_input->pose.orientation.z,
            pose_input->pose.position.x,
            pose_input->pose.position.y,
            pose_input->pose.position.z);

    bool has_result;
    has_result = depthmap_->add_frames(img_8uC1, T_world_curr.inv());
    if (has_result)
        denoiseAndPublishResults();
}

void quadmap::DepthmapNode::Msg_Callback_tf(
        const sensor_msgs::ImageConstPtr &image_input,
        const geometry_msgs::TransformStampedConstPtr &trans_input) {
    printf("\n\n\n");
    num_msgs_ += 1;
    current_msg_time = image_input->header.stamp;
    if (!depthmap_) {
        ROS_ERROR("depthmap not initialized. Call the DepthmapNode::init() method");
        return;
    }
    cv::Mat img_8uC1;
    try {
        cv_bridge::CvImageConstPtr cv_img_ptr =
                cv_bridge::toCvShare(image_input, sensor_msgs::image_encodings::MONO8);
        img_8uC1 = cv_img_ptr->image;
        img_8uC1.at<unsigned char>(0,0) = image_input->header.seq;
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    tf::StampedTransform imu_transform;
    tf::transformStampedMsgToTF(*trans_input, imu_transform);

    tf::Transform world_cam_tf;
    world_cam_tf = imu_transform * this->imu_cam_;

    quadmap::SE3<float> T_world_curr(
            world_cam_tf.getRotation().w(),
            world_cam_tf.getRotation().x(),
            world_cam_tf.getRotation().y(),
            world_cam_tf.getRotation().z(),
            world_cam_tf.getOrigin().x(),
            world_cam_tf.getOrigin().y(),
            world_cam_tf.getOrigin().z());

    bool has_result;
    has_result = depthmap_->add_frames(img_8uC1, T_world_curr.inv());
    if (has_result)
        denoiseAndPublishResults();
}

void quadmap::DepthmapNode::imageCb(const sensor_msgs::ImageConstPtr &image_input) {
    printf("\n\n");
    num_msgs_ += 1;
    current_msg_time = image_input->header.stamp;
    if (!depthmap_) {
        ROS_ERROR("depthmap not initialized. Call the DepthmapNode::init() method");
        return;
    }
    cv::Mat img_8uC1;
    try {
        cv_bridge::CvImageConstPtr cv_img_ptr =
                cv_bridge::toCvShare(image_input, sensor_msgs::image_encodings::MONO8);
        img_8uC1 = cv_img_ptr->image;
        img_8uC1.at<unsigned char>(0,0) = image_input->header.seq;
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    tf::StampedTransform transform;
    bool success{false};
    int count{0};
    while (!success) {
        try {
            tf_listener_.waitForTransform("world", tf_goal_frame_, current_msg_time, ros::Duration(3.0));
            tf_listener_.lookupTransform("world", tf_goal_frame_, current_msg_time, transform);
            success = true;
            ROS_INFO("Timestep: %lf", current_msg_time.toSec());
            break;
        } catch (tf::ExtrapolationException &e) {
            // avoid getting "Lookup would require extrapolation into the future." error
            ++count;
            ROS_INFO("Mismatch!");
        }
        if (count > 10)
            return;
        ros::Duration(0.01).sleep();
    }

    tf::Point pt = transform.getOrigin();
    tf::Quaternion rot = transform.getRotation();
    quadmap::SE3<float> T_world_curr(
            rot.w(),
            rot.x(),
            rot.y(),
            rot.z(),
            pt.x(),
            pt.y(),
            pt.z());

    bool has_result;
    has_result = depthmap_->add_frames(img_8uC1, T_world_curr.inv());
    if (has_result)
        denoiseAndPublishResults();
}

void quadmap::DepthmapNode::denoiseAndPublishResults() {
    std::async(std::launch::async,
               &quadmap::Publisher::publishDepthmapAndPointCloud,
               *publisher_,
               current_msg_time);
}

void quadmap::DepthmapNode::setFrameName(const std::string &frame_name) {
    tf_goal_frame_ = frame_name;
}

const std::string &quadmap::DepthmapNode::getFrameName() const {
    return tf_goal_frame_;
}

bool quadmap::DepthmapNode::setImuCam() {
    try {
        tf::TransformListener tmp_listener;
        tmp_listener.waitForTransform("/imu", "/cam02", ros::Time(0), ros::Duration(3.0));
        tmp_listener.lookupTransform("/imu", "/cam02", ros::Time(0), imu_cam_);
        return true;
    } catch (tf::TransformException &ex) {
        return false;}
}
