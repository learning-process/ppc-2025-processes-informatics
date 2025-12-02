#include "romanova_v_jacobi_method_processes/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "romanova_v_jacobi_method_processes/common/include/common.hpp"
#include "util/include/util.hpp"

namespace romanova_v_jacobi_method_processes {

RomanovaVJacobiMethodSEQ::RomanovaVJacobiMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool RomanovaVJacobiMethodSEQ::PreProcessingImpl() {
  return true;
}

bool RomanovaVJacobiMethodSEQ::ValidationImpl() {
  return true;
}

bool RomanovaVJacobiMethodSEQ::RunImpl() {
  return true;
}

bool RomanovaVJacobiMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace romanova_v_jacobi_method_processes
