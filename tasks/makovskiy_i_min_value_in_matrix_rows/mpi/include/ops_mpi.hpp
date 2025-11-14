#pragma once

#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

class MinValueMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MinValueMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace makovskiy_i_min_value_in_matrix_rows
