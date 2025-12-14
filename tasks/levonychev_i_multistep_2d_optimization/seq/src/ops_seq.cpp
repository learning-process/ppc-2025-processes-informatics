#include "levonychev_i_multistep_2d_optimization/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_multistep_2d_optimization {

LevonychevIMultistep2dOptimizationSEQ::LevonychevIMultistep2dOptimizationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LevonychevIMultistep2dOptimizationSEQ::ValidationImpl() {
  return true;
}

bool LevonychevIMultistep2dOptimizationSEQ::PreProcessingImpl() {
  return true;
}

bool LevonychevIMultistep2dOptimizationSEQ::RunImpl() {
  return true;
}

bool LevonychevIMultistep2dOptimizationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_multistep_2d_optimization
