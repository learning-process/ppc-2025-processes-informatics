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
  return (std::get<0>(GetInput()) > static_cast<size_t>(0));
}

bool RomanovACRSProductSEQ::PreProcessingImpl() {
  return true;
}

bool RomanovACRSProductSEQ::RunImpl() {
  // ====================================================== IMPL CODE ======================================================
  return true;
}

bool RomanovACRSProductSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_crs_product
