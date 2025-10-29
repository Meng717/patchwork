#ifndef INCLUDE_PATCHWORK_PATCHWORK_HPP_
#define INCLUDE_PATCHWORK_PATCHWORK_HPP_

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <boost/format.hpp>
#include <pcl/common/centroid.h>
#include <pcl/io/pcd_io.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "./zone_models.hpp"

int NOT_ASSIGNED = -2;
int FEW_POINTS = -1;
int UPRIGHT_ENOUGH = 0;     // cyan
int FLAT_ENOUGH = 1;        // green
int TOO_HIGH_ELEVATION = 2; // blue
int TOO_TILTED = 3;         // red
int GLOBALLY_TOO_HIGH_ELEVATION = 4;

// vector cout operator
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[ ";
  for (const auto &element : vec) {
    os << element << " ";
  }
  os << "]";
  return os;
}

#define NUM_HEURISTIC_MAX_PTS_IN_PATCH 3000
#define MARKER_Z_VALUE -2.2

using Eigen::JacobiSVD;
using Eigen::MatrixXf;
using Eigen::VectorXf;

using namespace std;

/*
    @brief PathWork ROS Node.
*/
template <typename PointT>
bool point_z_cmp(PointT a, PointT b) {
  return a.z < b.z;
}

struct PCAFeature {
  Eigen::Vector3f principal_;
  Eigen::Vector3f normal_;
  Eigen::Vector3f singular_values_;
  Eigen::Vector3f mean_;
  float d_;
  float th_dist_d_;
  float linearity_;
  float planarity_;
};

template <typename PointT>
struct Patch {
  bool is_close_to_origin_ = false;  // If so, we can set threshold more conservatively
  int ring_idx_ = NOT_ASSIGNED;
  int sector_idx_ = NOT_ASSIGNED;

  int status_ = NOT_ASSIGNED;

  PCAFeature feature_;

  pcl::PointCloud<PointT> cloud_;
  pcl::PointCloud<PointT> ground_;
  pcl::PointCloud<PointT> non_ground_;

  void clear() {
    if (!cloud_.empty()) cloud_.clear();
    if (!ground_.empty()) ground_.clear();
    if (!non_ground_.empty()) non_ground_.clear();
  }
};

template <typename PointT>
class PatchWork {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::vector<Patch<PointT>> Ring;
  typedef std::vector<Ring> RegionwisePatches;

  std::string frame_patchwork = "map";

  PatchWork(
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
      vector<double> flatness_thr
    )
  {

    sensor_height_ = sensor_height;
    sensor_model_ = sensor_model;

    ATAT_ON_ = ATAT_ON;
    max_r_for_ATAT_ = max_r_for_ATAT;
    num_sectors_for_ATAT_ = num_sectors_for_ATAT;
    noise_bound_ = noise_bound;

    num_iter_ = num_iter;
    num_lpr_ = num_lpr;
    num_min_pts_ = num_min_pts;
    th_seeds_ = th_seeds;
    th_dist_ = th_dist;
    max_range_ = max_range;
    min_range_ = min_range;  // It should cover the body size of the car.
    num_rings_ = num_rings;
    num_sectors_ = num_sectors;
    uprightness_thr_ = uprightness_thr;  // The larger, the more strict
    // The larger, the more soft
    adaptive_seed_selection_margin_ = adaptive_seed_selection_margin;

    // It is not in the paper
    // It is also not matched our philosophy, but it is employed to reject some
    // FPs easily & intuitively. For patchwork, it is only applied on Z3 and Z4
    using_global_thr_ = using_global_thr;
    global_elevation_thr_ = global_elevation_thr;

    // CZM denotes 'Concentric Zone Model'. Please refer to our paper
    // 2024.07.28. I feel `num_zones_`, `num_sectors_each_zone_`,
    // num_rings_each_zone_` are rarely fine-tuned. So I've decided to provide
    // predefined parameter sets for sensor types
    elevation_thr_ = elevation_thr;
    flatness_thr_ = flatness_thr;

    // It equals to elevation_thr_.size()/flatness_thr_.size();
    zone_model_ = ConcentricZoneModel(sensor_model_, sensor_height_, min_range_, max_range_);
    num_rings_of_interest_ = elevation_thr_.size();

    frame_patchwork, frame_patchwork;

    reverted_points_by_flatness_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);

    const auto &num_sectors_each_zone_ = zone_model_.sensor_config_.num_sectors_for_each_zone_;
    sector_sizes_ = {2 * M_PI / num_sectors_each_zone_.at(0),
                     2 * M_PI / num_sectors_each_zone_.at(1),
                     2 * M_PI / num_sectors_each_zone_.at(2),
                     2 * M_PI / num_sectors_each_zone_.at(3)};

    initialize(regionwise_patches_, zone_model_);
  }

  void estimate_ground(const pcl::PointCloud<PointT> &cloud_in,
                       pcl::PointCloud<PointT> &ground,
                       pcl::PointCloud<PointT> &nonground,
                       double &time_taken);

 private:
  // For ATAT (All-Terrain Automatic heighT estimator)
  bool ATAT_ON_;
  double noise_bound_;
  double max_r_for_ATAT_;
  int num_sectors_for_ATAT_;

  int num_iter_;
  int num_lpr_;
  int num_min_pts_;
  int num_rings_;
  int num_sectors_;
  int num_rings_of_interest_;

  double sensor_height_;
  double th_seeds_;
  double th_dist_;
  double max_range_;
  double min_range_;
  double uprightness_thr_;
  double adaptive_seed_selection_margin_;

  bool initialized_ = true;

  // For global threshold
  bool using_global_thr_;
  double global_elevation_thr_;

  // For visualization
  bool visualize_;

  std::string sensor_model_;
  vector<std::pair<int, int>> patch_indices_;  // For multi-threading. {ring_idx, sector_idx}
  ConcentricZoneModel zone_model_;

  vector<double> sector_sizes_;
  vector<double> ring_sizes_;
  vector<double> min_ranges_;
  vector<double> elevation_thr_;
  vector<double> flatness_thr_;

  RegionwisePatches regionwise_patches_;

  pcl::PointCloud<PointT> reverted_points_by_flatness_, rejected_points_by_elevation_;

  void initialize(RegionwisePatches &patches, const ConcentricZoneModel &zone_model);

  void flush_patches(RegionwisePatches &patches);

  double xy2theta(const double &x, const double &y);

  double xy2radius(const double &x, const double &y);

  void pc2regionwise_patches(const pcl::PointCloud<PointT> &src, RegionwisePatches &patches);

  void estimate_plane_(const pcl::PointCloud<PointT> &ground, PCAFeature &feat);

  void perform_regionwise_ground_segmentation(Patch<PointT> &patch,
                                              const bool is_h_available = true);

  double consensus_set_based_height_estimation(const Eigen::RowVectorXd &X,
                                               const Eigen::RowVectorXd &ranges,
                                               const Eigen::RowVectorXd &weights);

  void estimate_sensor_height(pcl::PointCloud<PointT> cloud_in);

  void extract_initial_seeds_(const pcl::PointCloud<PointT> &p_sorted,
                              pcl::PointCloud<PointT> &init_seeds,
                              bool is_close_to_origin,
                              bool is_h_available);

  int determine_ground_likelihood_estimation_status(const int concentric_idx,
                                                    const double z_vec,
                                                    const double z_elevation,
                                                    const double surface_variable);
};

template <typename PointT>
inline void PatchWork<PointT>::initialize(RegionwisePatches &patches,
                                          const ConcentricZoneModel &zone_model) {
  patches.clear();
  patch_indices_.clear();
  Patch<PointT> patch;

  // Reserve memory in advance to boost speed
  patch.cloud_.reserve(1000);
  patch.ground_.reserve(1000);
  patch.non_ground_.reserve(1000);

  // In polar coordinates, `num_columns` are `num_sectors`
  // and `num_rows` are `num_rings`, respectively
  int num_rows = zone_model_.num_total_rings_;
  const auto &num_sectors_per_ring = zone_model.num_sectors_per_ring_;

  for (int j = 0; j < num_rows; j++) {
    Ring ring;
    patch.ring_idx_ = j;
    patch.is_close_to_origin_ = j < zone_model.max_ring_index_in_first_zone;
    for (int i = 0; i < num_sectors_per_ring[j]; i++) {
      patch.sector_idx_ = i;
      ring.emplace_back(patch);

      patch_indices_.emplace_back(j, i);
    }
    patches.emplace_back(ring);
  }
}

template <typename PointT>
inline void PatchWork<PointT>::flush_patches(RegionwisePatches &patches) {
  int num_rows = patches.size();
  for (int j = 0; j < num_rows; j++) {
    int num_columns = patches[j].size();
    for (int i = 0; i < num_columns; i++) {
      patches[j][i].clear();
    }
  }
}

template <typename PointT>
inline void PatchWork<PointT>::estimate_plane_(const pcl::PointCloud<PointT> &ground,
                                               PCAFeature &feat) {
  Eigen::Vector4f pc_mean;
  Eigen::Matrix3f cov;
  pcl::computeMeanAndCovarianceMatrix(ground, cov, pc_mean);

  // Singular Value Decomposition: SVD
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
  feat.singular_values_ = svd.singularValues();

  feat.linearity_ =
      (feat.singular_values_(0) - feat.singular_values_(1)) / feat.singular_values_(0);
  feat.planarity_ =
      (feat.singular_values_(1) - feat.singular_values_(2)) / feat.singular_values_(0);

  // use the least singular vector as normal
  feat.normal_ = (svd.matrixU().col(2));
  if (feat.normal_(2) < 0) {  // z-direction of the normal vector should be positive
    feat.normal_ = -feat.normal_;
  }
  // mean ground seeds value
  feat.mean_ = pc_mean.head<3>();
  // according to normal.T*[x,y,z] = -d
  feat.d_ = -(feat.normal_.transpose() * feat.mean_)(0, 0);
  feat.th_dist_d_ = th_dist_ - feat.d_;
}

template <typename PointT>
inline void PatchWork<PointT>::extract_initial_seeds_(const pcl::PointCloud<PointT> &p_sorted,
                                                      pcl::PointCloud<PointT> &init_seeds,
                                                      bool is_close_to_origin,
                                                      bool is_h_available) {
  init_seeds.points.clear();

  // LPR is the mean of low point representative
  double sum = 0;
  int cnt = 0;

  int init_idx = 0;
  // Empirically, adaptive seed selection applying to Z1 is fine
  if (is_h_available) {
    static double lowest_h_margin_in_close_zone =
        (sensor_height_ == 0.0) ? -0.1 : adaptive_seed_selection_margin_ * sensor_height_;
    if (is_close_to_origin) {
      for (size_t i = 0; i < p_sorted.points.size(); i++) {
        if (p_sorted.points[i].z < lowest_h_margin_in_close_zone) {
          ++init_idx;
        } else {
          break;
        }
      }
    }
  }

  // Calculate the mean height value.
  for (size_t i = init_idx; i < p_sorted.points.size() && cnt < num_lpr_; i++) {
    sum += p_sorted.points[i].z;
    cnt++;
  }
  double lpr_height = cnt != 0 ? sum / cnt : 0;  // in case divide by 0

  // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
  for (size_t i = 0; i < p_sorted.points.size(); i++) {
    if (p_sorted.points[i].z < lpr_height + th_seeds_) {
      init_seeds.points.push_back(p_sorted.points[i]);
    }
  }
}

template <typename PointT>
inline double PatchWork<PointT>::consensus_set_based_height_estimation(
    const Eigen::RowVectorXd &X,
    const Eigen::RowVectorXd &ranges,
    const Eigen::RowVectorXd &weights) {
  // check input parameters - dimension inconsistent test
  assert(!((X.rows() != ranges.rows()) || (X.cols() != ranges.cols())));

  // check input parameters - only one element test
  assert(!((X.rows() == 1) && (X.cols() == 1)));

  int N = X.cols();
  std::vector<std::pair<double, int>> h;
  for (int i = 0; i < N; ++i) {
    h.push_back(std::make_pair(X(i) - ranges(i), i + 1));
    h.push_back(std::make_pair(X(i) + ranges(i), -i - 1));
  }

  // ascending order
  std::sort(h.begin(), h.end(), [](std::pair<double, int> a, std::pair<double, int> b) {
    return a.first < b.first;
  });

  int nr_centers = 2 * N;
  Eigen::RowVectorXd x_hat = Eigen::MatrixXd::Zero(1, nr_centers);
  Eigen::RowVectorXd x_cost = Eigen::MatrixXd::Zero(1, nr_centers);

  double ranges_inverse_sum = ranges.sum();
  double dot_X_weights = 0;
  double dot_weights_consensus = 0;
  int consensus_set_cardinal = 0;
  double sum_xi = 0;
  double sum_xi_square = 0;

  for (size_t i = 0; i < nr_centers; ++i) {
    int idx = static_cast<int>(std::abs(h.at(i).second)) - 1;  // Indices starting at 1
    int epsilon = (h.at(i).second > 0) ? 1 : -1;

    consensus_set_cardinal += epsilon;
    dot_weights_consensus += epsilon * weights(idx);
    dot_X_weights += epsilon * weights(idx) * X(idx);
    ranges_inverse_sum -= epsilon * ranges(idx);
    sum_xi += epsilon * X(idx);
    sum_xi_square += epsilon * X(idx) * X(idx);

    x_hat(i) = dot_X_weights / dot_weights_consensus;

    double residual =
        consensus_set_cardinal * x_hat(i) * x_hat(i) + sum_xi_square - 2 * sum_xi * x_hat(i);
    x_cost(i) = residual + ranges_inverse_sum;
  }

  size_t min_idx;
  x_cost.minCoeff(&min_idx);
  double estimate_temp = x_hat(min_idx);
  return estimate_temp;
}

template <typename PointT>
inline void PatchWork<PointT>::estimate_sensor_height(pcl::PointCloud<PointT> cloud_in) {
  // ATAT: All-Terrain Automatic HeighT estimator
  Ring ring_for_ATAT(num_sectors_for_ATAT_);
  for (auto const &pt : cloud_in.points) {
    int sector_idx;
    double r = xy2radius(pt.x, pt.y);

    float sector_size_for_ATAT = 2 * M_PI / num_sectors_for_ATAT_;

    if ((r <= max_r_for_ATAT_) && (r > min_range_)) {
      double theta = xy2theta(pt.x, pt.y);

      sector_idx = min(static_cast<int>((theta / sector_size_for_ATAT)), num_sectors_for_ATAT_);
      ring_for_ATAT[sector_idx].cloud_.points.emplace_back(pt);
    }
  }

  // Assign valid measurements and corresponding linearities/planarities
  vector<double> ground_elevations_wrt_the_origin;
  vector<double> linearities;
  vector<double> planarities;
  for (int i = 0; i < num_sectors_for_ATAT_; ++i) {
    if (static_cast<int>(ring_for_ATAT[i].cloud_.size()) < num_min_pts_) {
      continue;
    }

    pcl::PointCloud<PointT> dummy_est_ground;
    pcl::PointCloud<PointT> dummy_est_non_ground;
    perform_regionwise_ground_segmentation(ring_for_ATAT[i], false);

    const auto &feat = ring_for_ATAT[i].feature_;

    const double ground_z_vec = abs(feat.normal_(2));
    const double ground_z_elevation = feat.mean_(2);

    // Check whether the vector is sufficiently upright and flat
    if (ground_z_vec > uprightness_thr_ && feat.linearity_ < 0.9) {
      ground_elevations_wrt_the_origin.push_back(ground_z_elevation);
      linearities.push_back(feat.linearity_);
      planarities.push_back(feat.planarity_);
    }
  }

  // Setting for consensus set-based height estimation
  // It is equal to component-wise translation estimation (COTE) in TEASER++
  int N = ground_elevations_wrt_the_origin.size();

  if (ground_elevations_wrt_the_origin.size() == 0) {
    throw invalid_argument(
        "No valid ground points for ATAT! Please check the "
        "input data and `max_r_for_ATAT`");
  }
  Eigen::Matrix<double, 1, Eigen::Dynamic> values = Eigen::MatrixXd::Ones(1, N);
  Eigen::Matrix<double, 1, Eigen::Dynamic> ranges = noise_bound_ * Eigen::MatrixXd::Ones(1, N);
  Eigen::Matrix<double, 1, Eigen::Dynamic> weights =
      1.0 / (noise_bound_ * noise_bound_) * Eigen::MatrixXd::Ones(1, N);
  for (int i = 0; i < N; ++i) {
    values(0, i) = ground_elevations_wrt_the_origin[i];
    ranges(0, i) = ranges(0, i) * linearities[i];
    weights(0, i) = weights(0, i) * planarities[i] * planarities[i];
  }

  double estimated_h = consensus_set_based_height_estimation(values, ranges, weights);

  // Note that these are opposites
  sensor_height_ = -estimated_h;
}

template <typename PointT>
inline void PatchWork<PointT>::estimate_ground(const pcl::PointCloud<PointT> &cloud_in,
                                               pcl::PointCloud<PointT> &ground,
                                               pcl::PointCloud<PointT> &nonground,
                                               double &time_taken) {

  if (initialized_ && ATAT_ON_) {
    estimate_sensor_height(cloud_in);
    initialized_ = false;
  }

  // static double start, t0, t1, t2;
  // double                  t_total_ground   = 0.0;

  // 1.Msg to pointcloud
  pcl::PointCloud<PointT> cloud_in_tmp = cloud_in;

  static auto start = chrono::high_resolution_clock::now();

  // Error point removal
  // As there are some error mirror reflection under the ground,
  // Sort point according to height, here uses z-axis in default
  // -2.0 is a rough criteria
  size_t i = 0;
  while (i < cloud_in_tmp.points.size()) {
    if (cloud_in_tmp.points[i].z < -sensor_height_ - 2.0) {
      std::iter_swap(cloud_in_tmp.points.begin() + i, cloud_in_tmp.points.end() - 1);
      cloud_in_tmp.points.pop_back();
    } else {
      ++i;
    }
  }

  // t1 = ros::Time::now().toSec();
  flush_patches(regionwise_patches_);
  pc2regionwise_patches(cloud_in_tmp, regionwise_patches_);

  ground.clear();
  nonground.clear();
  reverted_points_by_flatness_.clear();
  rejected_points_by_elevation_.clear();

  int num_patches = patch_indices_.size();

  // HT comments: TBB w/ blocked_range was faster in my desktop
  // 117 Hz vs 126 Hz
  // tbb::parallel_for(0, num_patches, [&](int i) {
  tbb::parallel_for(tbb::blocked_range<int>(0, num_patches), [&](tbb::blocked_range<int> r) {
    for (int k = r.begin(); k < r.end(); ++k) {
      const auto &[ring_idx, sector_idx] = patch_indices_[k];
      auto &patch = regionwise_patches_[ring_idx][sector_idx];

      if (patch.cloud_.points.size() > num_min_pts_) {
        // 2022.05.02 update
        // Region-wise sorting is adopted, which is much faster than global
        // sorting!
        sort(patch.cloud_.points.begin(), patch.cloud_.points.end(), point_z_cmp<PointT>);
        perform_regionwise_ground_segmentation(patch);

        const auto &feat = patch.feature_;

        const double ground_z_vec = abs(feat.normal_(2));
        const double ground_z_elevation = feat.mean_(2);
        const double surface_variable =
            feat.singular_values_.minCoeff() /
            (feat.singular_values_(0) + feat.singular_values_(1) + feat.singular_values_(2));

        patch.status_ = determine_ground_likelihood_estimation_status(
            ring_idx, ground_z_vec, ground_z_elevation, surface_variable);
      } else {
        // Why? Because it is better to reject noise points
        // That is, these noise points sometimes lead to mis-recognition or
        // wrong clustering Thus, in practice, just rejecting them is better
        // than using them But note that this may degrade quantitative
        // ground segmentation performance
        patch.ground_ = patch.cloud_;
        patch.non_ground_.clear();
        patch.status_ = FEW_POINTS;
      }
    }
  });

  std::for_each(
      patch_indices_.begin(), patch_indices_.end(), [&](const std::pair<int, int> &index_pair) {
        int ring_idx = index_pair.first;
        int sector_idx = index_pair.second;

        const auto &patch = regionwise_patches_[ring_idx][sector_idx];

        const auto &feat = patch.feature_;
        const auto &regionwise_ground = patch.ground_;
        const auto &regionwise_nonground = patch.non_ground_;
        const auto status = patch.status_;

        if (status == FEW_POINTS) {
          ground += regionwise_ground;
          nonground += regionwise_nonground;
        } else if (status == TOO_TILTED) {
          // All points are rejected
          nonground += regionwise_ground;
          nonground += regionwise_nonground;
        } else if (status == GLOBALLY_TOO_HIGH_ELEVATION) {
          nonground += regionwise_ground;
          nonground += regionwise_nonground;
        } else if (status == TOO_HIGH_ELEVATION) {
          nonground += regionwise_ground;
          nonground += regionwise_nonground;
        } else if (status == FLAT_ENOUGH) {
          ground += regionwise_ground;
          nonground += regionwise_nonground;
        } else if (status == UPRIGHT_ENOUGH) {
          ground += regionwise_ground;
          nonground += regionwise_nonground;
        } else {
          std::invalid_argument(
              "Something wrong in "
              "`determine_ground_likelihood_estimation_status()` fn!");
        }
      });

  static auto end = chrono::high_resolution_clock::now();
  auto duration_s = std::chrono::duration<double, std::micro>(end - start);
  time_taken = duration_s.count();
  //    ofstream time_txt("/home/shapelim/patchwork_time_anal.txt",
  //    std::ios::app); time_txt<<t0 - start<<" "<<t1 - t0 <<" "<<t2-t1<<"
  //    "<<t_total_ground<< " "<<t_total_estimate<<"\n"; time_txt.close();

}

template <typename PointT>
inline double PatchWork<PointT>::xy2theta(const double &x,
                                          const double &y) {  // 0 ~ 2 * PI
  /*
    if (y >= 0) {
        return atan2(y, x); // 1, 2 quadrant
    } else {
        return 2 * M_PI + atan2(y, x);// 3, 4 quadrant
    }
  */
  auto atan_value = atan2(y, x);                               // EDITED!
  return atan_value > 0 ? atan_value : atan_value + 2 * M_PI;  // EDITED!
}

template <typename PointT>
inline double PatchWork<PointT>::xy2radius(const double &x, const double &y) {
  return sqrt(pow(x, 2) + pow(y, 2));
}

template <typename PointT>
inline void PatchWork<PointT>::pc2regionwise_patches(const pcl::PointCloud<PointT> &src,
                                                     RegionwisePatches &patches) {
  std::for_each(src.points.begin(), src.points.end(), [&](const auto &pt) {
    const auto &[ring_idx, sector_idx] = zone_model_.get_ring_sector_idx(pt.x, pt.y);
    if (ring_idx != INVALID_RING_IDX && ring_idx != OVERFLOWED_IDX) {
      patches[ring_idx][sector_idx].cloud_.points.emplace_back(pt);
    }
  });
}

// For adaptive
template <typename PointT>
inline void PatchWork<PointT>::perform_regionwise_ground_segmentation(Patch<PointT> &patch,
                                                                      const bool is_h_available) {
  // 0. Initialization
  int N = patch.cloud_.points.size();
  pcl::PointCloud<PointT> ground_tmp;
  ground_tmp.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);
  if (!patch.ground_.empty()) patch.ground_.clear();
  if (!patch.non_ground_.empty()) patch.non_ground_.clear();

  // 1. set seeds!
  extract_initial_seeds_(patch.cloud_, ground_tmp, patch.is_close_to_origin_, is_h_available);

  // 2. Extract ground
  for (int i = 0; i < num_iter_; i++) {
    estimate_plane_(ground_tmp, patch.feature_);
    ground_tmp.clear();

    // pointcloud to matrix
    Eigen::MatrixXf points(N, 3);
    int j = 0;
    for (auto &p : patch.cloud_.points) {
      points.row(j++) = p.getVector3fMap();
    }
    // ground plane model
    Eigen::VectorXf result = points * patch.feature_.normal_;
    // threshold filter
    for (int r = 0; r < N; r++) {
      if (i < num_iter_ - 1) {
        if (result[r] < patch.feature_.th_dist_d_) {
          ground_tmp.points.emplace_back(patch.cloud_[r]);
        }
      } else {  // Final stage
        if (result[r] < patch.feature_.th_dist_d_) {
          patch.ground_.points.emplace_back(patch.cloud_.points[r]);
        } else {
          patch.non_ground_.points.emplace_back(patch.cloud_.points[r]);
        }
      }
    }
  }
}

template <typename PointT>
inline int PatchWork<PointT>::determine_ground_likelihood_estimation_status(
    const int ring_idx,
    const double z_vec,
    const double z_elevation,
    const double surface_variable) {
  if (z_vec < uprightness_thr_) {
    return TOO_TILTED;
  } else {  // orthogonal
    if (ring_idx < num_rings_of_interest_) {
      if (z_elevation > -sensor_height_ + elevation_thr_[ring_idx]) {
        if (flatness_thr_[ring_idx] > surface_variable) {
          return FLAT_ENOUGH;
        } else {
          return TOO_HIGH_ELEVATION;
        }
      } else {
        return UPRIGHT_ENOUGH;
      }
    } else {
      if (using_global_thr_ && (z_elevation > global_elevation_thr_)) {
        return GLOBALLY_TOO_HIGH_ELEVATION;
      } else {
        return UPRIGHT_ENOUGH;
      }
    }
  }
}

#endif  // INCLUDE_PATCHWORK_PATCHWORK_HPP_
