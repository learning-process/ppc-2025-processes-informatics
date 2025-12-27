#include "romanov_a_crs_product/seq/include/ops_seq.hpp"

#include <cstddef>

#include "romanov_a_crs_product/common/include/common.hpp"

namespace romanov_a_crs_product {

RomanovACRSProductSEQ::RomanovACRSProductSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CRS(static_cast<size_t>(0));
}

bool RomanovACRSProductSEQ::ValidationImpl() {
  return (std::get<0>(GetInput()).size() == std::get<1>(GetInput()).size());
}

bool RomanovACRSProductSEQ::PreProcessingImpl() {
  return true;
}

bool RomanovACRSProductSEQ::RunImpl() {
    const CRS &A = std::get<0>(GetInput());
    const CRS &B = std::get<1>(GetInput());

    CRS C = A * B;

    GetOutput() = std::move(C);

    return true;
}

bool RomanovACRSProductSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_crs_product
