#include "levonychev_i_mult_matrix_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "levonychev_i_mult_matrix_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_mult_matrix_vec {

LevonychevIMultMatrixVecMPI::LevonychevIMultMatrixVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LevonychevIMultMatrixVecMPI::ValidationImpl() {
  return true;
}

bool LevonychevIMultMatrixVecMPI::PreProcessingImpl() {
  return true;
}

bool LevonychevIMultMatrixVecMPI::RunImpl() {
  return true;
}

bool LevonychevIMultMatrixVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_mult_matrix_vec
