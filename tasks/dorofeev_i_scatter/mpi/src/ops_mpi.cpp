#include "dorofeev_i_scatter/mpi/include/ops_mpi.hpp"

#include <vector>

#include "dorofeev_i_scatter/common/include/common.hpp"

namespace dorofeev_i_scatter {

DorofeevIScatterMPI::DorofeevIScatterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DorofeevIScatterMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool DorofeevIScatterMPI::PreProcessingImpl() {
  return true;
}

bool DorofeevIScatterMPI::RunImpl() {
  const auto &input = GetInput();
  double sum = 0.0;
  for (double val : input) {
    sum += val;
  }
  GetOutput() = sum;
  return true;
}

bool DorofeevIScatterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_scatter
