#include "spichek_d_simpson_integral/seq/include/ops_seq.hpp"

#include <cmath>

namespace spichek_d_simpson_integral {

SpichekDSimpsonIntegralSEQ::SpichekDSimpsonIntegralSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDSimpsonIntegralSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetInput() % 2 == 0);
}

bool SpichekDSimpsonIntegralSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDSimpsonIntegralSEQ::RunImpl() {
  int n = GetInput();
  double h = 1.0 / n;
  double sum = 0.0;

  for (int i = 0; i <= n; i++) {
    for (int j = 0; j <= n; j++) {
      double x = i * h;
      double y = j * h;

      int wx = (i == 0 || i == n) ? 1 : (i % 2 == 0 ? 2 : 4);
      int wy = (j == 0 || j == n) ? 1 : (j % 2 == 0 ? 2 : 4);

      sum += wx * wy * (x * x + y * y);
    }
  }

  double result = sum * h * h / 9.0;
  GetOutput() = static_cast<int>(std::round(result));
  return true;
}

bool SpichekDSimpsonIntegralSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_simpson_integral
