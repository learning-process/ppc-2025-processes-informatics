#include "kurpiakov_a_vert_tape_mat_vec_mul/seq/include/ops_seq.hpp"

#include <tuple>
#include <vector>

#include "kurpiakov_a_vert_tape_mat_vec_mul/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

KurpiakovAVretTapeMulSEQ::KurpiakovAVretTapeMulSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KurpiakovAVretTapeMulSEQ::ValidationImpl() {
  // int size_of_vec = std::get<2>(GetInput()).size();
  // int size_of_mat = std::get<1>(GetInput()).size();
  int size_of_input = std::get<0>(GetInput());
  // bool res = std::cmp_equal(size_of_vec / size_of_input, size_of_mat / (size_of_input * size_of_input));
  return (size_of_input >= 0);
}

bool KurpiakovAVretTapeMulSEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool KurpiakovAVretTapeMulSEQ::RunImpl() {
  const int total_size = std::get<0>(GetInput());

  if (total_size == 0) {
    GetOutput() = {0};
    return true;
  }

  OutType in_vec = std::get<2>(GetInput());
  OutType in_mat = std::get<1>(GetInput());

  OutType res_vec(total_size, 0);

  for (int i = 0; i < total_size; i++) {
    for (int j = 0; j < total_size; j++) {
      res_vec[i] += in_mat[i * total_size + j] * in_vec[j];
    }
  }

  GetOutput() = res_vec;
  return true;
}

bool KurpiakovAVretTapeMulSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
