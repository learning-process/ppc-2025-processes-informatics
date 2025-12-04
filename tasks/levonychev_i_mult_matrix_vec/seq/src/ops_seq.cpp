#include "levonychev_i_mult_matrix_vec/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "levonychev_i_mult_matrix_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_mult_matrix_vec {

LevonychevIMultMatrixVecSEQ::LevonychevIMultMatrixVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LevonychevIMultMatrixVecSEQ::ValidationImpl() {
  return true;
}

bool LevonychevIMultMatrixVecSEQ::PreProcessingImpl() {
  return true;
}

bool LevonychevIMultMatrixVecSEQ::RunImpl() {
  return true;
}

bool LevonychevIMultMatrixVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_mult_matrix_vec
