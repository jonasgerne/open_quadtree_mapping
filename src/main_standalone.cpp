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

void display(std::string&& name, cv::Mat& image) {
    cv::namedWindow(name.c_str(), cv::WINDOW_NORMAL);
    cv::resizeWindow(name.c_str(), image.cols, image.rows);
    cv::imshow(name.c_str(), image);
}

/**
 * Saves the image as a PFM file.
 * @brief savePFM
 * @param image
 * @param filePath
 * @return
 */
bool savePFM(const cv::Mat image, const std::string filePath)
{
    //Open the file as binary!
    std::ofstream imageFile(filePath.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

    if(imageFile)
    {
        int width(image.cols), height(image.rows);
        int numberOfComponents(image.channels());

        //Write the type of the PFM file and ends by a line return
        char type[3];
        type[0] = 'P';
        type[2] = 0x0a;

        if(numberOfComponents == 3)
        {
            type[1] = 'F';
        }
        else if(numberOfComponents == 1)
        {
            type[1] = 'f';
        }

        imageFile << type[0] << type[1] << type[2];

        //Write the width and height and ends by a line return
        imageFile << width << " " << height << type[2];

        //Assumes little endian storage and ends with a line return 0x0a
        //Stores the type
        char byteOrder[10];
        byteOrder[0] = '-'; byteOrder[1] = '1'; byteOrder[2] = '.'; byteOrder[3] = '0';
        byteOrder[4] = '0'; byteOrder[5] = '0'; byteOrder[6] = '0'; byteOrder[7] = '0';
        byteOrder[8] = '0'; byteOrder[9] = 0x0a;

        for(int i = 0 ; i<10 ; ++i)
        {
            imageFile << byteOrder[i];
        }

        //Store the floating points RGB color upside down, left to right
        float* buffer = new float[numberOfComponents];

        for(int i = 0 ; i<height ; ++i)
        {
            for(int j = 0 ; j<width ; ++j)
            {
                if(numberOfComponents == 1)
                {
                    buffer[0] = image.at<float>(height-1-i,j);
                }
                else
                {
                    cv::Vec3f color = image.at<cv::Vec3f>(height-1-i,j);

                    //OpenCV stores as BGR
                    buffer[0] = color.val[2];
                    buffer[1] = color.val[1];
                    buffer[2] = color.val[0];
                }

                //Write the values
                imageFile.write((char *) buffer, numberOfComponents*sizeof(float));

            }
        }

        delete[] buffer;

        imageFile.close();
    }
    else
    {
        std::cerr << "Could not open the file : " << filePath << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char **argv)
{
  if(!quadmap::checkCudaDevice(argc, argv))
    return EXIT_FAILURE;

  int semi2dense_ratio = 1; // 5
  int cost_downsampling = 1;
  bool doBeliefPropagation = true;
  bool useQuadtree = false;
  float P1 = 0.003f;
  float P2 = 0.01f;

  // Distortion coefficients
  float k1 = 0.0f;
  float k2 = 0.0f;
  float r1 = 0.0f;
  float r2 = 0.0f;

  // Read intrinsics
  std::string intrinsicsPath = argv[1];
  std::ifstream intrinFile(intrinsicsPath);
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
  std::string posesPath = argv[2];
  std::ifstream posesFile(posesPath);
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
  std::string rgbPattern = argv[3];
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

  std::shared_ptr<quadmap::Depthmap> depthmap_ = std::make_shared<quadmap::Depthmap>(width, height, cost_downsampling, fx, cx, fy, cy, undist_map1, undist_map2, semi2dense_ratio, doBeliefPropagation, useQuadtree, P1, P2);

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

      std::cout << "T_world_curr" << std::endl << T_world_curr << std::endl;
      std::cout << "T_world_curr inv" << std::endl << T_world_curr.inv() << std::endl;

      bool has_result;
      has_result = depthmap_->add_frames(img_8uC1, T_world_curr.inv());
      if (has_result) {
          double minVal = 0.0, maxVal = 0.0;
          cv::Mat reference_mat = depthmap_->getReferenceImage();
          cv::Mat keyframe_mat = depthmap_->getKeyframeImage();
          cv::Mat keyframe;
          cv::cvtColor(keyframe_mat, keyframe, cv::COLOR_GRAY2BGR);
          cv::Mat depthmap_mat = depthmap_->getDepthmap();
          sprintf(buffer, "%010d.pfm", idx);
          savePFM(depthmap_mat, buffer);
          cv::minMaxIdx(depthmap_mat, &minVal, &maxVal);
          
          static cv::Mat depthNorm, depthColor;
          float minDepth = 1.0f;
          float maxDepth = 50.0f;
          cv::threshold(depthmap_mat, depthmap_mat, minDepth, minDepth, cv::THRESH_TOZERO);
          cv::threshold(depthmap_mat, depthmap_mat, maxDepth, maxDepth, cv::THRESH_TRUNC);
          cv::normalize(depthmap_mat, depthNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
          cv::applyColorMap(depthNorm, depthColor, cv::COLORMAP_JET);
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
          display("Keyframe", keyframe);
          display("Reference", reference_mat);
          display("Depth", depthColor);
          display("Debug", debug_mat);
          //display("Epipolar", epipolar);
          cv::waitKey(1);
      }
  }

  return EXIT_SUCCESS;
}
