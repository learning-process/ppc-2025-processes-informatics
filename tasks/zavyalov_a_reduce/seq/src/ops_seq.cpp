#include <mpi.h>

#include <cstdlib>
#include <vector>

#include "zavyalov_a_reduce/common/include/common.hpp"
#include "zavyalov_a_reduce/seq/include/ops_seq.hpp"

namespace zavyalov_a_reduce {

ZavyalovAReduceSEQ::ZavyalovAReduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = nullptr;
}

bool ZavyalovAReduceSEQ::ValidationImpl() {
  return true;
  // return (!std::get<0>(GetInput()).empty()) && (std::get<0>(GetInput()).size() == std::get<1>(GetInput()).size());
}

bool ZavyalovAReduceSEQ::PreProcessingImpl() {
  return true;
}

bool ZavyalovAReduceSEQ::RunImpl() {

  return true;
}

bool ZavyalovAReduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_reduce
