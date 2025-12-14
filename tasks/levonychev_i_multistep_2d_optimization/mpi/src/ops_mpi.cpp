#include "levonychev_i_multistep_2d_optimization/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_multistep_2d_optimization {

LevonychevIMultistep2dOptimizationMPI::LevonychevIMultistep2dOptimizationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LevonychevIMultistep2dOptimizationMPI::ValidationImpl() {
  return true;
}

bool LevonychevIMultistep2dOptimizationMPI::PreProcessingImpl() {
  return true;
}

bool LevonychevIMultistep2dOptimizationMPI::RunImpl() {
  return true;
}

bool LevonychevIMultistep2dOptimizationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_multistep_2d_optimization
