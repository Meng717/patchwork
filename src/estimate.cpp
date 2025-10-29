//
// Created by Hyungtae Lim on 6/23/21.
//

// For disable PCL complile lib, to use PointXYZILID
#define PCL_NO_PRECOMPILE
#include <signal.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdlib>

#include "patchwork.hpp"

using PointType = pcl::PointXYZ;
using namespace std;
namespace py = pybind11;

boost::shared_ptr<PatchWork<PointType>> PatchworkGroundSeg;

void numpy_to_pcd(const py::array_t<float> &input, pcl::PointCloud<PointType> &cloud)
{
  auto buf = input.request();
  if (buf.ndim != 2 || buf.shape[1] != 3)
  {
    throw std::runtime_error("Input must be a (N, 3) float32 array");
  }

  size_t num_points = buf.shape[0];
  const float *ptr = static_cast<float *>(buf.ptr);

  cloud.clear();
  cloud.width = num_points;
  cloud.height = 1;
  cloud.is_dense = false;
  cloud.points.resize(num_points);

  for (size_t i = 0; i < num_points; ++i)
  {
    cloud.points[i].x = ptr[3 * i + 0];
    cloud.points[i].y = ptr[3 * i + 1];
    cloud.points[i].z = ptr[3 * i + 2];
  }
}

py::array_t<float> pcd_to_numpy(const pcl::PointCloud<PointType> &cloud)
{
  size_t num_points = cloud.points.size();
  py::array_t<float> output(std::vector<size_t>{num_points, 3});
  auto buf = output.mutable_unchecked<2>();

  for (size_t i = 0; i < num_points; ++i)
  {
    buf(i, 0) = cloud.points[i].x;
    buf(i, 1) = cloud.points[i].y;
    buf(i, 2) = cloud.points[i].z;
  }

  return output;
}

py::array_t<float> removeRoadPoints(
    const py::array_t<float> &input,

    bool ATAT_ON,
    double noise_bound,
    double max_r_for_ATAT,
    int num_sectors_for_ATAT,

    int num_iter,
    int num_lpr,
    int num_min_pts,
    int num_rings,
    int num_sectors,

    double sensor_height,
    double th_seeds,
    double th_dist,
    double max_range,
    double min_range,
    double uprightness_thr,
    double adaptive_seed_selection_margin,

    bool using_global_thr,
    double global_elevation_thr,

    string sensor_model,
    vector<double> elevation_thr,
    vector<double> flatness_thr)
{

  PatchworkGroundSeg.reset(new PatchWork<PointType>(
    ATAT_ON, noise_bound, max_r_for_ATAT, num_sectors_for_ATAT, num_iter, num_lpr, num_min_pts, 
    num_rings, num_sectors, sensor_height, th_seeds, th_dist, 
    max_range, min_range, uprightness_thr, adaptive_seed_selection_margin, using_global_thr, 
    global_elevation_thr, sensor_model, elevation_thr, flatness_thr
  ));

  pcl::PointCloud<PointType> pc_curr;
  numpy_to_pcd(input, pc_curr);

  pcl::PointCloud<PointType> pc_ground;
  pcl::PointCloud<PointType> pc_non_ground;

  static double time_taken;

  PatchworkGroundSeg->estimate_ground(pc_curr, pc_ground, pc_non_ground, time_taken);
  auto output = pcd_to_numpy(pc_non_ground);
  PatchworkGroundSeg.reset();
  return output;
}

PYBIND11_MODULE(patchwork, m)
{
  m.def("removeRoadPoints", &removeRoadPoints);
}