#include "smyshlaev_a_gauss_filt/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "smyshlaev_a_gauss_filt/common/include/common.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_gauss_filt {

SmyshlaevAGaussFiltMPI::SmyshlaevAGaussFiltMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SmyshlaevAGaussFiltMPI::ValidationImpl() {
  return true;
}

bool SmyshlaevAGaussFiltMPI::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAGaussFiltMPI::RunImpl() {
  return true;
}

bool SmyshlaevAGaussFiltMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_gauss_filt
