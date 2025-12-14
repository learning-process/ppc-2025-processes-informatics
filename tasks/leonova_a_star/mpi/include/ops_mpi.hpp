#pragma once

#include "leonova_a_star/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_star {

class LeonovaAStarMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LeonovaAStarMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool ValidateMatricesOnMaster();  // Убрать const и static

  static std::vector<std::vector<int>> MultiplyMatricesMpi(const std::vector<std::vector<int>> &matrix_a,
                                                           const std::vector<std::vector<int>> &matrix_b);

  void BroadcastResult(int rank);
};

}  // namespace leonova_a_star
