#pragma once

#include "romanova_v_jacobi_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanova_v_jacobi_method {

class RomanovaVJacobiMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RomanovaVJacobiMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool isDiagonallyDominant(const std::vector<OutType>& matrix);
  bool isConverge(const OutType& prev, const OutType& curr);

  OutType x_;
  OutType A_; //чтобы не мучаться с пересылкой будем сразу хранить данные в одномерном массиве
  OutType b_;
  double eps_{};
  size_t maxIterations_{};

  OutType local_data_;

  size_t n_{};
  size_t st_row_{};
  size_t local_n_{};
};

}  // namespace romanova_v_jacobi_method
