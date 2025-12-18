#include "shvetsova_k_gausse_vert_strip/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"

namespace shvetsova_k_gausse_vert_strip {

ShvetsovaKGaussVertStripMPI::ShvetsovaKGaussVertStripMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>{0};
}

bool ShvetsovaKGaussVertStripMPI::ValidationImpl() {
  return true;
}

bool ShvetsovaKGaussVertStripMPI::PreProcessingImpl() {
  return true;
}

bool ShvetsovaKGaussVertStripMPI::RunImpl() {
  return true;
}

bool ShvetsovaKGaussVertStripMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_gausse_vert_strip
