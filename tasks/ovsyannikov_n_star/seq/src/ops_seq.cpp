#include "ovsyannikov_n_star/seq/include/ops_seq.hpp"

#include <vector>

namespace ovsyannikov_n_star {

OvsyannikovNStarSEQ::OvsyannikovNStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OvsyannikovNStarSEQ::ValidationImpl() {
  return GetInput().size() == 3;
}

bool OvsyannikovNStarSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool OvsyannikovNStarSEQ::RunImpl() {
  GetOutput() = GetInput()[2];
  return true;
}

bool OvsyannikovNStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_star
