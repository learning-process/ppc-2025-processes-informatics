#include "romanov_a_integration_rect_method/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include <cmath>

#include "romanov_a_integration_rect_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace romanov_a_integration_rect_method {

RomanovAIntegrationRectMethodSEQ::RomanovAIntegrationRectMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool RomanovAIntegrationRectMethodSEQ::ValidationImpl() {
  return (std::get<3>(GetInput()) > 0) && (IsEqual(GetOutput(), 0.0));
}

bool RomanovAIntegrationRectMethodSEQ::PreProcessingImpl() {
  return true;
}

bool RomanovAIntegrationRectMethodSEQ::RunImpl() {
  auto [f, a, b, n] = GetInput();

  double sgn = 1.0;
  if (a > b) {
    std::swap(a, b);
    sgn = -1.0;
  }

  double delta_x = (b - a) / static_cast<double>(n);

  double mid = a + delta_x / 2.0;

  for (int i = 0; i < n; ++i) {
    GetOutput() += f(mid) * delta_x;
    mid += delta_x;
  }

  GetOutput() *= sgn;

  return true;
}

bool RomanovAIntegrationRectMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_integration_rect_method
