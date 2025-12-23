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

 void GenerateGridPoints(std::vector<Point>& grid_points, double x_min, double x_max, double y_min, double y_max, int grid_size);
  [[nodiscard]] static std::vector<Point> SelectTopCandidates(const std::vector<Point> &points, int num_candidates);
  void BuildNewRegionsFromCandidates(const std::vector<Point> &candidates, int step, std::vector<SearchRegion> &new_regions);
  Point ApplyLocalOptimizationToCandidates(const std::vector<Point> &candidates);
  void SetFinalResult(const std::vector<Point> &candidates);
};

}  // namespace levonychev_i_multistep_2d_optimization
