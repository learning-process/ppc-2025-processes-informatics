#include "smyshlaev_a_mat_mul/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "smyshlaev_a_mat_mul/common/include/common.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_mat_mul {

SmyshlaevAMatMulMPI::SmyshlaevAMatMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SmyshlaevAMatMulMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool SmyshlaevAMatMulMPI::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAMatMulMPI::RunImpl() {
  
  return true;
}

bool SmyshlaevAMatMulMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_mat_mul
