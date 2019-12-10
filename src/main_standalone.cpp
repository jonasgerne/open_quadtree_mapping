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

#include <fstream>
#include <opencv2/opencv.hpp>

#include <quadmap/check_cuda_device.cuh>
#include <quadmap/depthmap.h>
#include <quadmap/se3.cuh>

#define cimg_plugin1 "cvMat.h"
#include "CImg.h"

void display(std::string&& name, cv::Mat& image) {
    cv::namedWindow(name.c_str(), cv::WINDOW_NORMAL);
    cv::resizeWindow(name.c_str(), image.cols, image.rows);
    cv::imshow(name.c_str(), image);
}

int main(int argc, char **argv)
{
  if(!quadmap::checkCudaDevice(argc, argv))
    return EXIT_FAILURE;

  // Additional parameters
  // frameelement.cuh:9 KEYFRAME_NUM
  // stereo_parameter.cuh:all
  // pixel_cost.cuh:4 DEPTH_NUM
  // depth_fusion.cuh

  bool display_enabled = false;
  bool fixNearPoint = false;
  bool printTimings = false;
  float P1 = 0.003f; // 0.003 (original)
  float P2 = 0.01f; // 0.01 (original)

  // Distortion coefficients
  float k1 = 0.0f;
  float k2 = 0.0f;
  float r1 = 0.0f;
  float r2 = 0.0f;

  // Arguments
  if(argc < 20){
      printf("Not enough parameters specified!\n");
      return -1;
  }
  std::string intrinsicsPath = argv[1];
  std::string posesPath = argv[2];
  std::string rgbPattern = argv[3];
  bool doBeliefPropagation = atoi(argv[4]);
  bool useQuadtree = atoi(argv[5]);
  bool doFusion = atoi(argv[6]);
  bool doGlobalUpsampling = atoi(argv[7]);
  int semi2dense_ratio = atoi(argv[8]); // 5 (original)
  int cost_downsampling = atoi(argv[9]); // 4 (original)
  float min_inlier_ratio_good = atof(argv[10]); // 0.6 (original)
  float min_inlier_ratio_bad = atof(argv[11]); // 0.45 (original)
  float prev_variance_factor = atof(argv[12]); // 1.0f (original)
  float new_variance_factor = atof(argv[13]); // 1.0f (original)
  float variance_offset = atof(argv[14]); // 0.0f (original)
  float minDepth = atof(argv[15]);
  float maxDepth = atof(argv[16]);
  bool inverse_depth = atoi(argv[17]);
  float new_keyframe_max_distance = atof(argv[18]);
  float new_keyframe_max_angle = cos(atof(argv[19]) / 180.0f * M_PI);

  float new_reference_max_distance = new_keyframe_max_distance; // 0.03; (original)
  float new_reference_max_angle = new_keyframe_max_angle; // 0.95f; (original)

  // Read intrinsics
  std::ifstream intrinFile(intrinsicsPath);
  if(!intrinFile.good()){
      printf("Could not open intrinsics file %s\n", intrinsicsPath.c_str());
      return -1;
  }
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> intrinsics;
  std::string line;
  while (std::getline(intrinFile, line))
  {
      //std::istringstream iss(line);
      // The matrix is saved row-wise in the file
      /*float p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23;
      iss >> p00 >> p01 >> p02 >> p03 >> p10 >> p11 >> p12 >> p13 >> p20 >> p21 >> p22 >> p23;*/
      float k00 = 0.0f, k01 = 0.0f, k02 = 0.0f, k10 = 0.0f, k11 = 0.0f, k12 = 0.0f, k20 = 0.0f, k21 = 0.0f, k22 = 0.0f;
      //iss >> k00, k01, k02, k10, k11, k12, k20, k21, k22;
      sscanf(line.c_str(), "%f %f %f %f %f %f %f %f %f", &k00, &k01, &k02, &k10, &k11, &k12, &k20, &k21, &k22);
      intrinsics << k00, k01, k02, k10, k11, k12, k20, k21, k22;
  }

  float fx = intrinsics(0, 0);
  float fy = intrinsics(1, 1);
  float cx = intrinsics(0, 2);
  float cy = intrinsics(1, 2);

  // Read poses
  std::ifstream posesFile(posesPath);
  if(!posesFile.good()){
      printf("Could not open poses file %s\n", posesPath.c_str());
      return -1;
  }
  std::vector<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> > poses;

  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> firstPoseInv;
  firstPoseInv.setIdentity();

  while (std::getline(posesFile, line))
  {
      std::istringstream iss(line);
      // The matrix is saved row-wise in the file
      float p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23;
      iss >> p00 >> p01 >> p02 >> p03 >> p10 >> p11 >> p12 >> p13 >> p20 >> p21 >> p22 >> p23;

      Eigen::Vector3f t;
      t << p03, p13, p23;
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R;
      R << p00, p01, p02, p10, p11, p12, p20, p21, p22;

      Eigen::Matrix<float, 4, 4, Eigen::RowMajor> pose;
      pose.setIdentity();
      //pose.block<3, 3>(0, 0) = R.transpose();
      //pose.block<3, 1>(0, 3) = -pose.block<3, 3>(0, 0) * t;
      pose.block<3, 3>(0, 0) = R;
      pose.block<3, 1>(0, 3) = t;

      if (poses.size() == 0) {
          firstPoseInv = pose.inverse();
      }

      pose = firstPoseInv * pose;

      poses.push_back(pose);
  }

  // Initialize image size
  char buffer[255];
  sprintf(buffer, rgbPattern.c_str(), 0);
  cv::Mat imgRaw = cv::imread(buffer, cv::IMREAD_GRAYSCALE);
  int height = imgRaw.rows;
  int width = imgRaw.cols;

  // initial the remap mat, it is used for undistort and also resive the image
  cv::Mat input_K = (cv::Mat_<float>(3, 3) << fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f);
  cv::Mat input_D = (cv::Mat_<float>(1, 4) << k1, k2, r1, r2);

  cv::Mat undist_map1, undist_map2;
  cv::initUndistortRectifyMap(
      input_K,
      input_D,
      cv::Mat_<double>::eye(3, 3),
      input_K,
      cv::Size(width, height),
      CV_32FC1,
      undist_map1, undist_map2);

  std::shared_ptr<quadmap::Depthmap> depthmap_ = std::make_shared<quadmap::Depthmap>(width, height, cost_downsampling,
          fx, cx, fy, cy, undist_map1, undist_map2, semi2dense_ratio, doBeliefPropagation, useQuadtree, doFusion,
          doGlobalUpsampling, fixNearPoint, printTimings, P1, P2, inverse_depth, minDepth, maxDepth, new_keyframe_max_angle, new_keyframe_max_distance, new_reference_max_angle,
          new_reference_max_distance, min_inlier_ratio_good, min_inlier_ratio_bad, new_variance_factor, prev_variance_factor, variance_offset);

  // Run
  for (int idx = 0; idx < poses.size(); idx++)
  {
      // Read image
      sprintf(buffer, rgbPattern.c_str(), idx);
      cv::Mat img_8uC1 = cv::imread(buffer, cv::IMREAD_GRAYSCALE);

      // Convert Pose
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R = poses[idx].block<3, 3>(0, 0);
      Eigen::Vector3f t = poses[idx].block<3, 1>(0, 3);
      quadmap::SE3<float> T_world_curr(R.data(), t.data());

      //std::cout << "T_world_curr" << std::endl << T_world_curr << std::endl;
      //std::cout << "T_world_curr inv" << std::endl << T_world_curr.inv() << std::endl;

      bool has_result;
      has_result = depthmap_->add_frames(img_8uC1, T_world_curr.inv());
      if (has_result) {
          double minVal = 0.0, maxVal = 0.0;
          cv::Mat reference_mat = depthmap_->getReferenceImage();
          cv::Mat keyframe_mat = depthmap_->getKeyframeImage();
          cv::Mat keyframe;
          cv::cvtColor(keyframe_mat, keyframe, cv::COLOR_GRAY2BGR);
          cv::Mat depthmap_mat = depthmap_->getDepthmap();
          cv::minMaxIdx(depthmap_mat, &minVal, &maxVal);
          sprintf(buffer, "%010d.pfm", idx);
          cimg_library::CImg<float> depth(depthmap_mat);
          depth.save(buffer);

          cv::minMaxIdx(depthmap_mat, &minVal, &maxVal);
          
          cv::Mat depthNorm, depthColor;
          //cv::threshold(depthmap_mat, depthmap_mat, minDepth, minDepth, cv::THRESH_TOZERO);
          //cv::threshold(depthmap_mat, depthmap_mat, maxDepth, maxDepth, cv::THRESH_TRUNC);
          //cv::normalize(depthmap_mat, depthNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
          //cv::applyColorMap(depthNorm, depthColor, cv::COLORMAP_JET);
          depthNorm = depthmap_mat.clone();
          depthNorm = (depthNorm - minDepth) / (maxDepth - minDepth) * 255.0f;
          depthNorm.convertTo(depthNorm, CV_8U);
          cv::applyColorMap(depthNorm, depthColor, cv::COLORMAP_JET);
          for (int i = 0; i < depthmap_mat.rows; i++) {
              for (int j = 0; j < depthmap_mat.cols; j++) {
                  if (std::isnan(depthmap_mat.at<float>(i, j))) {
                      depthColor.at<cv::Vec3b>(i, j)[0] = 0;
                      depthColor.at<cv::Vec3b>(i, j)[1] = 0;
                      depthColor.at<cv::Vec3b>(i, j)[2] = 0;
                  }
                  if (depthmap_mat.at<float>(i, j) < minDepth) {
                      depthColor.at<cv::Vec3b>(i, j)[0] = 0;
                      depthColor.at<cv::Vec3b>(i, j)[1] = 0;
                      depthColor.at<cv::Vec3b>(i, j)[2] = 0;
                  }
                  if (depthmap_mat.at<float>(i, j) > maxDepth) {
                      depthColor.at<cv::Vec3b>(i, j)[0] = 0;
                      depthColor.at<cv::Vec3b>(i, j)[1] = 0;
                      depthColor.at<cv::Vec3b>(i, j)[2] = 0;
                  }
              }
          }

          sprintf(buffer, "%010d.png", idx);
          cv::imwrite(buffer, depthColor);
          cv::Mat debug_mat = depthmap_->getDebugmap();

         /* cv::Mat epipolar_mat = depthmap_->getEpipolarImage();
          cv::Mat epipolar;
          cv::cvtColor(img_8uC1, epipolar, cv::COLOR_GRAY2BGR);
          int x = 120;
          int y = 175;
          cv::circle(keyframe, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
          cv::Vec4f& ep = epipolar_mat.at<cv::Vec4f>(y, x);
          printf("Epipolar | Near %f, %f | Far %f, %f", ep(0), ep(1), ep(2), ep(3));
          cv::line(epipolar, cv::Point(ep(0), ep(1)), cv::Point(ep(2), ep(3)), cv::Scalar(0, 0, 255), 3);*/
          /*int step = 50;
          for (int i = 0; i < height; i++) {
              for (int j = 0; j < width; j++) {
                  cv::Vec4f& ep = epipolar_mat.at<cv::Vec4f>(i, j);
                  if ((ep(0) >= 0) && ((i % step) == 0) && ((j % step) == 0)) {
                      cv::line(epipolar, cv::Point(ep(0), ep(1)), cv::Point(ep(2), ep(3)), cv::Scalar(0, 0, 255), 3);
                  }
              }
          }*/
          if(display_enabled) {
              display("Keyframe", keyframe);
              display("Reference", reference_mat);
              display("Depth", depthColor);
              display("Debug", debug_mat);
              //display("Epipolar", epipolar);
              cv::waitKey(1);
          }
      }
  }

  return EXIT_SUCCESS;
}
