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

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Eigen>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <quadmap/check_cuda_device.cuh>
#include <quadmap/depthmap.h>
#include <quadmap/se3.cuh>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <direct.h>
#endif

namespace py = pybind11;

std::shared_ptr<quadmap::Depthmap> depthmap_;

bool initialize(Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K, int width, int height, int cost_downsampling,
        bool doBeliefPropagation, bool useQuadtree, bool doFusion, bool doGlobalUpsampling, bool fixNearPoint, bool printTimings,
        float P1, float P2, float new_keyframe_max_angle, float new_keyframe_max_distance, float new_reference_max_angle,
        float new_reference_max_distance, float min_inlier_ratio_good, float min_inlier_ratio_bad) {

    int semi2dense_ratio = 5;

    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);

    // Distortion coefficients
    float k1 = 0.0f;
    float k2 = 0.0f;
    float r1 = 0.0f;
    float r2 = 0.0f;

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

    depthmap_ = std::make_shared<quadmap::Depthmap>(width, height, cost_downsampling, fx, cx, fy, cy, undist_map1,
            undist_map2, semi2dense_ratio, doBeliefPropagation, useQuadtree, doFusion, doGlobalUpsampling, fixNearPoint, printTimings,
            P1, P2, new_keyframe_max_angle, new_keyframe_max_distance, new_reference_max_angle, new_reference_max_distance,
            min_inlier_ratio_good, min_inlier_ratio_bad);

    return true;
}

py::tuple compute(py::array_t<unsigned char, py::array::c_style | py::array::forcecast> I, py::array_t<float, py::array::c_style | py::array::forcecast> T) {
    int nImages = I.shape(0);
    int height = I.shape(1);
    int width = I.shape(2);

    unsigned char* imgPtr = (unsigned char*)I.mutable_data();
    float* posePtr = (float*)T.mutable_data();

    for (int i = 0; i < nImages; i++) {
        // Image
        cv::Mat img = cv::Mat(height, width, CV_8UC1, &imgPtr[i*width*height]);

        // Pose
        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Te = Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> >(&posePtr[i*4*4]);
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R = Te.block<3, 3>(0, 0);
        Eigen::Vector3f t = Te.block<3, 1>(0, 3);
        quadmap::SE3<float> T_world_curr(R.data(), t.data());

        std::cout << "T_world_curr" << std::endl << T_world_curr << std::endl;
        std::cout << "T_world_curr inv" << std::endl << T_world_curr.inv() << std::endl;

        depthmap_->add_frames(img, T_world_curr.inv());
    }

    py::array_t<float> depth({ height, width });
        
    cv::Mat depthMap = depthmap_->getDepthmap();

    /*
    double minVal = 0.0, maxVal = 0.0;
    cv::Mat reference_mat = depthmap_->getReferenceImage();
    cv::Mat keyframe_mat = depthmap_->getKeyframeImage();
    cv::Mat keyframe;
    cv::cvtColor(keyframe_mat, keyframe, cv::COLOR_GRAY2BGR);
    static cv::Mat depthNorm, depthColor;
    float minDepth = 1.0f;
    float maxDepth = 50.0f;
    cv::threshold(depthmap_mat, depthmap_mat, minDepth, minDepth, cv::THRESH_TOZERO);
    cv::threshold(depthmap_mat, depthmap_mat, maxDepth, maxDepth, cv::THRESH_TRUNC);
    cv::normalize(depthmap_mat, depthNorm, 0, 255, CV_MINMAX, CV_8U);
    cv::applyColorMap(depthNorm, depthColor, cv::COLORMAP_JET);
    cv::Mat debug_mat = depthmap_->getDebugmap();
    cv::Mat epipolar_mat = depthmap_->getEpipolarImage();
    cv::Mat epipolar;
    cv::cvtColor(img_8uC1, epipolar, cv::COLOR_GRAY2BGR);
        
    int step = 50;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec4f& ep = epipolar_mat.at<cv::Vec4f>(i, j);
            if ((ep(0) >= 0) && ((i % step) == 0) && ((j % step) == 0)) {
                cv::line(epipolar, cv::Point(ep(0), ep(1)), cv::Point(ep(2), ep(3)), cv::Scalar(0, 0, 255), 3);
            }
        }
    }

    int x = 120;
    int y = 175;
    cv::circle(keyframe, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
    cv::Vec4f& ep = epipolar_mat.at<cv::Vec4f>(y, x);
    printf("Epipolar | Near %f, %f | Far %f, %f", ep(0), ep(1), ep(2), ep(3));
    cv::line(epipolar, cv::Point(ep(0), ep(1)), cv::Point(ep(2), ep(3)), cv::Scalar(0, 0, 255), 3);
    cv::imshow("Keyframe", keyframe);
    cv::imshow("Reference", reference_mat);
    cv::imshow("Depth", depthColor);
    cv::imshow("Debug", debug_mat);
    cv::imshow("Epipolar", epipolar);
    cv::waitKey();
    */

    // Write result
    float* depthPtr = (float*)depth.mutable_data();
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++)
        {
            int center = i + width * j;
            depthPtr[center] = depthMap.at<float>(center);
        }

    return py::make_tuple(depth);
}

// wrap as Python module
PYBIND11_MODULE(pyQuadtreeMapping, m)
{
    m.doc() = "quadtree mapping python wrapper";

    m.def("initialize", &initialize);
    m.def("compute", &compute);
}
