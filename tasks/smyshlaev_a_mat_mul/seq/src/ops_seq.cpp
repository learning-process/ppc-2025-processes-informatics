#include "smyshlaev_a_mat_mul/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "smyshlaev_a_mat_mul/common/include/common.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_mat_mul {

SmyshlaevAMatMulSEQ::SmyshlaevAMatMulSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SmyshlaevAMatMulSEQ::ValidationImpl() {
  return true;
}

bool SmyshlaevAMatMulSEQ::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAMatMulSEQ::RunImpl() {

  return true;
}

bool SmyshlaevAMatMulSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_mat_mul
