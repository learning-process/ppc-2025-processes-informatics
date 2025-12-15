#pragma once

#include <cstdint>
#include <vector>

#include "kurpiakov_a_vert_tape_mat_vec_mul/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

class KurpiakovAVretTapeMulMPI : public ppc::task::Task<InType, OutType> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KurpiakovAVretTapeMulMPI(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int64_t matrix_size_{0};
  std::vector<int64_t> matrix_data_;
  std::vector<int64_t> vector_data_;
  std::vector<int64_t> result_;

  std::vector<int64_t> local_vector_;
  std::vector<int64_t> local_matrix_;
  std::vector<int64_t> local_result_;
  int local_cols_{0};

  int rank_{-1};
  int world_size_{-1};
};

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
