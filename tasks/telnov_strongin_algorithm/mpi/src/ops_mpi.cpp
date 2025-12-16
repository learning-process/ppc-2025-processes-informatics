#include "telnov_strongin_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "telnov_strongin_algorithm/common/include/common.hpp"

namespace telnov_strongin_algorithm {

struct MaxData {
  double value{};
  int index{};
};

TelnovStronginAlgorithmMPI::TelnovStronginAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TelnovStronginAlgorithmMPI::ValidationImpl() {
  const auto &in = GetInput();
  return (in.eps > 0.0) && (in.b > in.a);
}

bool TelnovStronginAlgorithmMPI::PreProcessingImpl() {
  return true;
}

bool TelnovStronginAlgorithmMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &in = GetInput();
  const double eps = in.eps;

  auto f = [](double x) { return ((x - 1.0) * (x - 1.0)) + 1.0; };

  std::vector<double> x_vals{in.a, in.b};
  std::vector<double> f_vals{f(in.a), f(in.b)};

  const double r = 2.0;
  const int k_max_iters = 100;
  int iter = 0;

  while ((x_vals.back() - x_vals.front()) > eps && iter < k_max_iters) {
    ++iter;

    double m = 0.0;
    for (std::size_t i = 1; i < x_vals.size(); ++i) {
      const double dx = x_vals[i] - x_vals[i - 1];
      const double df = std::abs(f_vals[i] - f_vals[i - 1]);
      m = std::max(m, df / dx);
    }
    m = (m == 0.0) ? 1.0 : m;

    MaxData local_data;
    local_data.value = -1e9;
    local_data.index = 1;

    for (std::size_t i = static_cast<std::size_t>(rank) + 1; i < x_vals.size(); i += static_cast<std::size_t>(size)) {
      const double dx = x_vals[i] - x_vals[i - 1];
      const double df = f_vals[i] - f_vals[i - 1];

      const double r_val = (r * dx) + ((df * df) / (r * dx)) - (2.0 * (f_vals[i] + f_vals[i - 1]));

      if (r_val > local_data.value) {
        local_data.value = r_val;
        local_data.index = static_cast<int>(i);
      }
    }

    MaxData global_data{};
    MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (rank != 0) {
      int n = 0;
      MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
      x_vals.resize(static_cast<std::size_t>(n));
      f_vals.resize(static_cast<std::size_t>(n));
      MPI_Bcast(x_vals.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(f_vals.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      continue;
    }

    const int idx = global_data.index;
    double new_x = (0.5 * (x_vals[idx] + x_vals[idx - 1])) - ((f_vals[idx] - f_vals[idx - 1]) / (2.0 * m));

    new_x = (new_x <= x_vals[idx - 1] || new_x >= x_vals[idx]) ? (0.5 * (x_vals[idx] + x_vals[idx - 1])) : new_x;

    x_vals.insert(x_vals.begin() + idx, new_x);
    f_vals.insert(f_vals.begin() + idx, f(new_x));

    int n = static_cast<int>(x_vals.size());
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(x_vals.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(f_vals.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  GetOutput() = *std::ranges::min_element(f_vals);
  return true;
}

bool TelnovStronginAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace telnov_strongin_algorithm
