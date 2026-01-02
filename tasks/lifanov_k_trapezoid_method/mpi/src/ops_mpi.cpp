#include "lifanov_k_trapezoid_method/mpi/include/ops_mpi.hpp"

#include <cmath>

#include "lifanov_k_trapezoid_method/common/include/common.hpp"

#include <mpi.h>

namespace lifanov_k_trapezoid_method {

// Интегрируемая функция
static double f(double x, double y) {
  return x * x + y * y;  // пример
}

LifanovKTrapezoidMethodMPI::LifanovKTrapezoidMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool LifanovKTrapezoidMethodMPI::ValidationImpl() {
  // Ожидаем: a, b, c, d, nx, ny
  if (GetInput().size() != 6) {
    return false;
  }
  if (GetInput()[4] <= 0 || GetInput()[5] <= 0) {
    return false;
  }
  return true;
}

bool LifanovKTrapezoidMethodMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool LifanovKTrapezoidMethodMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Чтение входных данных
  const double a = GetInput()[0];
  const double b = GetInput()[1];
  const double c = GetInput()[2];
  const double d = GetInput()[3];
  const int nx = static_cast<int>(GetInput()[4]);
  const int ny = static_cast<int>(GetInput()[5]);

  const double hx = (b - a) / nx;
  const double hy = (d - c) / ny;

  // Разбиение по x
  const int base = nx / size;
  const int rem = nx % size;

  const int i_start = rank * base + std::min(rank, rem);
  const int i_end = i_start + base - 1 + (rank < rem ? 1 : 0);

  double local_sum = 0.0;

  for (int i = i_start; i <= i_end; ++i) {
    double x = a + i * hx;
    double wx = (i == 0 || i == nx) ? 0.5 : 1.0;

    for (int j = 0; j <= ny; ++j) {
      double y = c + j * hy;
      double wy = (j == 0 || j == ny) ? 0.5 : 1.0;

      local_sum += wx * wy * f(x, y);
    }
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_sum * hx * hy;
  }

  return true;
}

bool LifanovKTrapezoidMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lifanov_k_trapezoid_method
