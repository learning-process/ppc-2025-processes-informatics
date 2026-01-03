#include "dorofeev_i_scatter/seq/include/ops_seq.hpp"

#include <vector>

#include "dorofeev_i_scatter/common/include/common.hpp"

namespace dorofeev_i_scatter {

DorofeevIScatterSEQ::DorofeevIScatterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DorofeevIScatterSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool DorofeevIScatterSEQ::PreProcessingImpl() {
  return true;
}

bool DorofeevIScatterSEQ::RunImpl() {
  const auto &input = GetInput();
  double sum = 0.0;
  for (double val : input) {
    sum += val;
  }
  GetOutput() = sum;
  return true;
}

bool DorofeevIScatterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_scatter
