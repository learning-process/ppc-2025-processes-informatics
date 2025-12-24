#include "zenin_a_gauss_filter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "zenin_a_gauss_filter/common/include/common.hpp"

namespace zenin_a_gauss_filter {

ZeninAGaussFilterMPI::ZeninAGaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninAGaussFilterMPI::ValidationImpl() {
  return true;
}

bool ZeninAGaussFilterMPI::PreProcessingImpl() {
  return true;
}

bool ZeninAGaussFilterMPI::RunImpl() {
  return true;
}

bool ZeninAGaussFilterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_gauss_filter
