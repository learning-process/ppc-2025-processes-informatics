#pragma once

#include <cstddef>
#include <vector>

#include "kondakov_v_global_search/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondakov_v_global_search {

class KondakovVGlobalSearchSEQ : public ppc::task::Task<InType, OutType> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KondakovVGlobalSearchSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double EvaluateFunction(double x);

  std::vector<double> points_x_;
  std::vector<double> values_y_;
  double best_point_ = 0.0;
  double best_value_ = 0.0;
  int total_evals_ = 0;
  bool has_converged_ = false;

  [[nodiscard]] double ComputeAdaptiveLipschitzEstimate(double r) const;
  [[nodiscard]] double IntervalMerit(std::size_t i, double l_est) const;
  [[nodiscard]] double ProposeTrialPoint(std::size_t i, double l_est) const;
  [[nodiscard]] std::size_t LocateInsertionIndex(double x) const;
  void InsertEvaluation(double x, double fx);
};

}  // namespace kondakov_v_global_search
