#ifndef RANSAC_HPP
#define RANSAC_HPP

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <algorithm>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include <fstream>
#include <ctime>
#include <chrono>
#include <random>
#include <cmath>

struct PlaneModel {
    Eigen::Vector4d coefficients;           // a, b, c, d for the plane equation ax + by + cz + d = 0
    std::vector<Eigen::Vector3d> inliers;   // Points that are within the threshold distance from the plane
    float std_dev;                          // Standard deviation of the inliers

    // Constructor to initialize members
    PlaneModel() : coefficients(Eigen::Vector4d::Zero()), std_dev(0.0f) {}

    // Member function to fit a plane using RANSAC
};

struct PointCloudData {
    Eigen::Vector3f position; // Stores (x, y, z)
    Eigen::Vector3f color;    // Stores (r, g, b)

    PointCloudData(float x, float y, float z, float r = 1.0f, float g = 1.0f, float b = 1.0f)
        : position(x, y, z), color(r, g, b) {}
};


PlaneModel fit_plane_ransac(const std::vector<PointCloudData>& data, float threshold = 0.01, int max_iterations = 100, const Eigen::Vector3d& remove = Eigen::Vector3d(0, 1, 0));
// Free functions
Eigen::Vector3d calculate_plane_normal(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3);
float angle_between_vectors(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);

#endif // RANSAC_HPP