#include "romanov_a_crs_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>

#include "romanov_a_crs_product/common/include/common.hpp"

namespace romanov_a_crs_product {

RomanovACRSProductMPI::RomanovACRSProductMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CRS(static_cast<size_t>(0));
}

bool RomanovACRSProductMPI::ValidationImpl() {
  return (std::get<0>(GetInput()).getCols() == std::get<1>(GetInput()).getRows());
}

bool RomanovACRSProductMPI::PreProcessingImpl() {
  return true;
}

bool RomanovACRSProductMPI::RunImpl() {
  const CRS &A = std::get<0>(GetInput());
  const CRS &B = std::get<1>(GetInput());

  CRS C = A * B;

  GetOutput() = std::move(C);

  return true;
}

bool RomanovACRSProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_crs_product
