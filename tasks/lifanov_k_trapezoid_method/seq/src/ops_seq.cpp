#include "lifanov_k_trapezoid_method/seq/include/ops_seq.hpp"

#include <cmath>

#include "lifanov_k_trapezoid_method/common/include/common.hpp"

namespace lifanov_k_trapezoid_method {

// Интегрируемая функция (та же, что в MPI)
static double f(double x, double y) {
  return x * x + y * y;  // пример
}

LifanovKTrapezoidMethodSEQ::LifanovKTrapezoidMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool LifanovKTrapezoidMethodSEQ::ValidationImpl() {
  // Ожидаем: a, b, c, d, nx, ny
  if (GetInput().size() != 6) {
    return false;
  }
  if (GetInput()[4] <= 0 || GetInput()[5] <= 0) {
    return false;
  }
  return true;
}

bool LifanovKTrapezoidMethodSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool LifanovKTrapezoidMethodSEQ::RunImpl() {
  const double a = GetInput()[0];
  const double b = GetInput()[1];
  const double c = GetInput()[2];
  const double d = GetInput()[3];
  const int nx = static_cast<int>(GetInput()[4]);
  const int ny = static_cast<int>(GetInput()[5]);

  const double hx = (b - a) / nx;
  const double hy = (d - c) / ny;

  double sum = 0.0;

  for (int i = 0; i <= nx; ++i) {
    const double x = a + i * hx;
    const double wx = (i == 0 || i == nx) ? 0.5 : 1.0;

    for (int j = 0; j <= ny; ++j) {
      const double y = c + j * hy;
      const double wy = (j == 0 || j == ny) ? 0.5 : 1.0;

      sum += wx * wy * f(x, y);
    }
  }

  GetOutput() = sum * hx * hy;
  return true;
}

bool LifanovKTrapezoidMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lifanov_k_trapezoid_method
