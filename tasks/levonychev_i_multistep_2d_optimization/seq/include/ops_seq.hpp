#pragma once

#include <vector>

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "levonychev_i_multistep_2d_optimization/common/include/optimization_common.hpp"
#include "task/include/task.hpp"

namespace levonychev_i_multistep_2d_optimization {

class LevonychevIMultistep2dOptimizationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LevonychevIMultistep2dOptimizationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<Point> GenerateGridPoints(double x_min, double x_max, double y_min, double y_max, int grid_size);
  std::vector<Point> SelectTopCandidates(const std::vector<Point> &points, int num_candidates) const;
  void UpdateSearchRegionFromCandidates(const std::vector<Point> &candidates, double &x_min, double &x_max,
                                        double &y_min, double &y_max);
  Point ApplyLocalOptimizationToCandidates(const std::vector<Point> &candidates);
  void SetFinalResult(const std::vector<Point> &candidates);
};

}  // namespace levonychev_i_multistep_2d_optimization
