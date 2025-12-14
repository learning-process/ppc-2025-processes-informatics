#include "pylaeva_s_simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace pylaeva_s_simple_iteration_method {

PylaevaSSimpleIterationMethodMPI::PylaevaSSimpleIterationMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PylaevaSSimpleIterationMethodMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool PylaevaSSimpleIterationMethodMPI::PreProcessingImpl() {
  
  return GetOutput() > 0;
}

bool PylaevaSSimpleIterationMethodMPI::RunImpl() {
  auto input = GetInput();



  MPI_Barrier(MPI_COMM_WORLD);
  return GetOutput() > 0;
}

bool PylaevaSSimpleIterationMethodMPI::PostProcessingImpl() {
  
  return GetOutput() > 0;
}

}  // namespace pylaeva_s_simple_iteration_method
