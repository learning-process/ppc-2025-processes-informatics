#pragma once

#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sizov_d_global_search {

class SizovDGlobalSearchSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SizovDGlobalSearchSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  using Seed = std::pair<double, double>;

  bool CollectInitialSeeds(const InType &problem, std::vector<Seed> &seeds) const;
  void ExpandSeedsAroundBest(const InType &problem, double uniq_eps, std::vector<Seed> &seeds) const;
  void InitializeStateFromSeeds(const std::vector<Seed> &seeds);

  std::optional<double> EvaluatePoint(const InType &problem, double point, bool enforce_limit) const;
  bool SeekFiniteSample(const InType &problem, double start, double direction, double &point, double &value) const;

  double DynamicLimit(double sample) const;
  double ClampValue(double value, bool *clipped) const;
  double EstimateM(double reliability) const;
  std::pair<double, bool> EvaluateIntervalCandidate(const InType &problem, double candidate, double left_point,
                                                    double right_point, double left_value, double right_value) const;
  double FilterCandidateValue(double candidate_value, double left_value, double right_value, bool *clipped) const;
  bool TryInsertPoint(double point, double value, double insert_eps);

  std::vector<double> points_;
  std::vector<double> values_;
  std::vector<char> clipped_;
  double best_point_ = 0.0;
  double best_value_ = 0.0;
  int iterations_ = 0;
  bool converged_ = false;
  double value_ceiling_ = 1e6;
  double negative_ceiling_ = 1e6;
};

}  // namespace sizov_d_global_search
