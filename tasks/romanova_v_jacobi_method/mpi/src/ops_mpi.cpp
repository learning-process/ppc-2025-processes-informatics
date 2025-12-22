#include "romanova_v_jacobi_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "romanova_v_jacobi_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace romanova_v_jacobi_method {

RomanovaVJacobiMethodMPI::RomanovaVJacobiMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput();
}

bool RomanovaVJacobiMethodMPI::ValidationImpl() {
  return true;
}

bool RomanovaVJacobiMethodMPI::PreProcessingImpl() {
  return true;
}

bool RomanovaVJacobiMethodMPI::RunImpl() {
  return true;
}

bool RomanovaVJacobiMethodMPI::PostProcessingImpl() {
  return true;
}

}  // romanova_v_jacobi_method
