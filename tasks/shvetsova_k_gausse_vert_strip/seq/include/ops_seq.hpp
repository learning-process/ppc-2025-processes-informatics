#pragma once

#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_gausse_vert_strip {

class ShvetsovaKGaussVertStripSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShvetsovaKGaussVertStripSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные методы для снижения когнитивной сложности
  void FindPivotAndSwap(int target_row, int n, std::vector<std::vector<double>> &band, std::vector<int> &offsets,
                        std::vector<double> &vec) const;

  void EliminateBelow(int target_row, int n, std::vector<std::vector<double>> &band, const std::vector<int> &offsets,
                      std::vector<double> &vec) const;

  int size_of_rib_ = 0;
};

}  // namespace shvetsova_k_gausse_vert_strip
