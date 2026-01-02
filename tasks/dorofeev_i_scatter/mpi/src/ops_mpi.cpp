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
  // Placeholder: just take the first element like seq
  GetOutput() = GetInput()[0];
  return true;
}

bool DorofeevIScatterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_scatter
