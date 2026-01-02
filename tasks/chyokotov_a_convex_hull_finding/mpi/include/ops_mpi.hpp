#pragma once

#include <utility>
#include <vector>

#include "chyokotov_a_convex_hull_finding/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chyokotov_a_convex_hull_finding {

class ChyokotovConvexHullFindingMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChyokotovConvexHullFindingMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool Check(const std::vector<std::vector<int>> &input);
  std::pair<std::vector<int>, int> DistributeImageData(int rank, int size, int width, int height);
  std::pair<int, int> CalculateRowDistribution(int rank, int size, int height);
  long long Cross(const std::pair<int, int> &a, const std::pair<int, int> &b, const std::pair<int, int> &c);
  std::vector<std::vector<std::pair<int, int>>> FindConnectedComponentsMPI(int rank, int size, int start_row,
                                                                           int end_row, int width, int height,
                                                                           const std::vector<int> &local_data);

  std::vector<std::pair<int, int>> ConvexHullAndrew(const std::vector<std::pair<int, int>> &points);

  void ExchangeBoundaryRows(bool has_top, bool has_bottom, int rank, int size, int width,
                            const std::vector<int> &local_pixels, std::vector<int> &extended_pixels);

  std::vector<std::vector<std::pair<int, int>>> ProcessExtendedRegion(const std::vector<int> &extended_pixels,
                                                                      int extended_rows, int width,
                                                                      int global_y_offset);

  std::vector<std::pair<int, int>> ExtractComponent(int start_x, int start_y, const std::vector<int> &extended_pixels,
                                                    std::vector<std::vector<bool>> &visited, int width,
                                                    int extended_rows, int global_y_offset);

  std::vector<std::vector<std::pair<int, int>>> FilterLocalComponents(
      const std::vector<std::vector<std::pair<int, int>>> &all_components, int start_row, int end_row);

  void SendHullsToRank0(const std::vector<std::vector<std::pair<int, int>>> &local_hulls);
  void ReceiveHullsFromRank(int src_rank, std::vector<int> &all_sizes, std::vector<int> &global_flat);

  std::vector<std::vector<std::pair<int, int>>> GatherHullsOnRank0(
      int rank, int size, const std::vector<std::vector<std::pair<int, int>>> &local_hulls);

  std::vector<std::vector<std::pair<int, int>>> BroadcastResultToAllRanks(
      int rank, const std::vector<std::vector<std::pair<int, int>>> &global_hulls_on_rank0);
};

}  // namespace chyokotov_a_convex_hull_finding
