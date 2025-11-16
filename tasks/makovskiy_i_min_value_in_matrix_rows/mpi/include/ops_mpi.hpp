#pragma once

#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"
#include "task/include/task.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

class MinValueMPI : public BaseTask {
 public:
  // ---> ИСПРАВЛЕНИЕ: Заменено '.' на '::' <---
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MinValueMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные функции
  void ProcessRankZero(std::vector<int> &local_min_values);
  void GatherResults(const std::vector<int> &local_min_values);

  // Статическая вспомогательная функция, так как она не зависит от состояния объекта
  static void ProcessWorkerRank(std::vector<int> &local_min_values);
};

}  // namespace makovskiy_i_min_value_in_matrix_rows
