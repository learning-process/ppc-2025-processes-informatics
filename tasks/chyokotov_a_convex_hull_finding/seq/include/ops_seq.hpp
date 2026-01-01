#pragma once

#include "chyokotov_a_convex_hull_finding/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chyokotov_a_convex_hull_finding {

class ChyokotovConvexHullFindingSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ChyokotovConvexHullFindingSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<std::vector<std::pair<int, int>>> FindComponent();
  int Cross(const std::pair<int, int> &o, const std::pair<int, int> &a, const std::pair<int, int> &b);
  std::vector<std::pair<int, int>> ConvexHull(std::vector<std::pair<int, int>> x);
};

}  // namespace chyokotov_a_convex_hull_finding
