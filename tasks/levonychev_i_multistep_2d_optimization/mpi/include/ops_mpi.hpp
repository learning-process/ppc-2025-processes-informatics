#pragma once
#include <vector>

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "levonychev_i_multistep_2d_optimization/common/include/optimization_common.hpp"
#include "task/include/task.hpp"

namespace levonychev_i_multistep_2d_optimization {

class LevonychevIMultistep2dOptimizationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LevonychevIMultistep2dOptimizationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void InitializeRegions(int rank, int size, const OptimizationParams &params, SearchRegion &my_region);

  static void ExecuteOptimizationSteps(int rank, int size, const OptimizationParams &params, SearchRegion &my_region,
                                       OptimizationResult &result);

  static void GatherAndSelectCandidates(int rank, const std::vector<Point> &local_candidates,
                                        std::vector<Point> &all_candidates);

  static void ScatterNewRegions(int rank, int size, const OptimizationParams &params,
                                const std::vector<Point> &all_candidates, int step, SearchRegion &my_region);

  static void BuildNewRegions(const OptimizationParams &params, const std::vector<Point> &all_candidates, int step,
                              std::vector<SearchRegion> &new_regions);

  static void DistributeRegionsToProcesses(int rank, int size, const OptimizationParams &params,
                                           const std::vector<SearchRegion> &new_regions, SearchRegion &my_region);

  static Point FindGlobalBest(int rank, int size, const OptimizationParams &params, const SearchRegion &my_region);

  static void GatherFinalPoints(int rank, int size, const Point &my_best, std::vector<Point> &all_final_points);

  static Point SelectGlobalBest(int rank, const OptimizationParams &params, const std::vector<Point> &all_final_points);

  static void ProcessLocalCandidates(const OptimizationParams &params, const std::vector<Point> &local_points,
                                     std::vector<Point> &local_candidates);
};

}  // namespace levonychev_i_multistep_2d_optimization
