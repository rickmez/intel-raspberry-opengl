#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <filesystem>
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
#include "ransac/ransac.hpp"
#include <limits>

// Calculate the angle between two vectors
float angle_between_vectors(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
    float dot_product = v1.dot(v2);
    float magnitudes = v1.norm() * v2.norm();
    if (magnitudes < 1e-6) { // Avoid division by zero
        return 0.0f;
    }
    return std::acos(dot_product / magnitudes);
}

// Calculate the normal vector of a plane defined by three points
Eigen::Vector3d calculate_plane_normal(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3) {
    Eigen::Vector3d v1 = p2 - p1;
    Eigen::Vector3d v2 = p3 - p1;
    Eigen::Vector3d normal = v1.cross(v2);
    normal.normalize();
    return normal;
}

// Fit a plane using RANSAC


PlaneModel fit_plane_ransac(const std::vector<PointCloudData>& data, float threshold, int max_iterations, const Eigen::Vector3d& remove) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    int bestSupport = 0;
    PlaneModel best_plane;
    best_plane.std_dev = std::numeric_limits<float>::max();

    for (int i = 0; i < max_iterations; ++i) {
        // Randomly sample three points and cast to double if needed.
        Eigen::Vector3d p1 = data[dis(gen)].position.cast<double>();
        Eigen::Vector3d p2 = data[dis(gen)].position.cast<double>();
        Eigen::Vector3d p3 = data[dis(gen)].position.cast<double>();

        // Fit a plane to the sampled points.
        Eigen::Vector3d normalVector = calculate_plane_normal(p1, p2, p3);
        float diff = angle_between_vectors(normalVector, remove) * (180.0f / M_PI);

        if (diff < 10) {
            float d = -normalVector.dot(p1);
            Eigen::Vector4d Currplane;
            Currplane << normalVector, d;

            // Calculate distances from all points to the plane
            std::vector<float> distances(data.size());
            for (size_t j = 0; j < data.size(); ++j) {
                // Convert each point to double precision before calculating the distance
                distances[j] = std::abs(normalVector.dot(data[j].position.cast<double>()) + d) / normalVector.norm();
            }

            // Count inliers (points within the threshold distance from the plane)
            std::vector<Eigen::Vector3d> inliers;
            for (size_t j = 0; j < data.size(); ++j) {
                if (distances[j] < threshold) {
                    inliers.push_back(data[j].position.cast<double>());
                }
            }

            if (inliers.empty()) continue; // Avoid division by zero

            // Calculate standard deviation of inliers
            float mean = 0;
            for (const auto& inlier : inliers) {
                mean += normalVector.dot(inlier) + d;
            }
            mean /= inliers.size();

            float std_dev = 0;
            for (const auto& inlier : inliers) {
                std_dev += std::pow(normalVector.dot(inlier) + d - mean, 2);
            }
            std_dev = std::sqrt(std_dev / inliers.size());

            // Update best plane if the current one has more inliers or a better standard deviation
            if (inliers.size() > bestSupport || (inliers.size() == bestSupport && std_dev < best_plane.std_dev)) {
                bestSupport = inliers.size();
                best_plane.coefficients = Currplane;
                best_plane.inliers = std::move(inliers);
                best_plane.std_dev = std_dev;
            }
        }
    }

    return best_plane;
}
