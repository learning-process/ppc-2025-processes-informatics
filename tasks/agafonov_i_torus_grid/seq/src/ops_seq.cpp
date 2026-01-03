#include "agafonov_i_torus_grid/seq/include/ops_seq.hpp"

#include "agafonov_i_torus_grid/common/include/common.hpp"

namespace agafonov_i_torus_grid {

TorusGridTaskSEQ::TorusGridTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TorusGridTaskSEQ::ValidationImpl() {
  return true;
}

bool TorusGridTaskSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TorusGridTaskSEQ::RunImpl() {
  GetOutput() = GetInput().value;
  return true;
}

bool TorusGridTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace agafonov_i_torus_grid
