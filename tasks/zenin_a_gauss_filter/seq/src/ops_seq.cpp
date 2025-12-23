#include "zenin_a_gauss_filter/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

#include "zenin_a_gauss_filter/common/include/common.hpp"

namespace zenin_a_gauss_filter {

ZeninAGaussFilterSEQ::ZeninAGaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninAGaussFilterSEQ::ValidationImpl() {
  const auto &in = GetInput();

  return !std::get<2>(in).empty();
}

bool ZeninAGaussFilterSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ZeninAGaussFilterSEQ::RunImpl() {
  const auto &in = GetInput();
  const auto &data = std::get<2>(in);

  GetOutput() = data;
  return true;
}

bool ZeninAGaussFilterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_gauss_filter
