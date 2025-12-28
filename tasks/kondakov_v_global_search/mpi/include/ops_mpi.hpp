#pragma once

#include <cstddef>
#include <vector>

#include "kondakov_v_global_search/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondakov_v_global_search {

class KondakovVGlobalSearchMPI : public ppc::task::Task<InType, OutType> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KondakovVGlobalSearchMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double EvaluateFunction(double x);

  [[nodiscard]] bool IsRoot() const;
  [[nodiscard]] double ComputeAdaptiveLipschitzEstimate(double r) const;
  [[nodiscard]] double IntervalMerit(std::size_t i, double l_est) const;
  [[nodiscard]] double ProposeTrialPoint(std::size_t i, double l_est) const;
  [[nodiscard]] std::size_t LocateInsertionIndex(double x) const;
  void SyncGlobalState();
  void InsertEvaluation(double x, double fx);

  void SelectIntervalsToRefine(double l_est, std::vector<std::pair<double, std::size_t>> &merits);
  bool CheckConvergence(const Params &cfg, const std::vector<std::pair<double, std::size_t>> &merits);
  void GatherAndBroadcastTrialResults(const std::vector<std::pair<double, std::size_t>> &merits, int num_trials,
                                      double l_est);

  std::vector<double> points_x_;
  std::vector<double> values_y_;
  double best_point_ = 0.0;
  double best_value_ = 0.0;
  int total_evals_ = 0;
  bool has_converged_ = false;
  int world_rank_ = 0;
  int world_size_ = 1;
};

}  // namespace kondakov_v_global_search
