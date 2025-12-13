#include "telnov_strongin_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <vector>

#include "telnov_strongin_algorithm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace telnov_strongin_algorithm {

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
  const double a = in.a;
  const double b = in.b;
  const double eps = in.eps;

  auto f = [](double x) { return ((x - 1.0) * (x - 1.0)) + 1.0; };

  std::vector<double> x_vals = {a, b};
  std::vector<double> f_vals = {f(a), f(b)};

  while ((x_vals.back() - x_vals.front()) > eps) {
    double m = 0.0;
    for (std::size_t i = 1; i < x_vals.size(); ++i) {
      m = std::max(m, std::abs(f_vals[i] - f_vals[i - 1]) / (x_vals[i] - x_vals[i - 1]));
    }

    if (m == 0.0) {
      m = 1.0;
    }

    const double r = 2.0;
    double local_max_r = -1e9;
    int local_idx = 1;

    for (std::size_t i = static_cast<std::size_t>(rank + 1); i < x_vals.size(); i += static_cast<std::size_t>(size)) {
      const double dx = x_vals[i] - x_vals[i - 1];
      const double df = f_vals[i] - f_vals[i - 1];
      const double r_val = (r * dx) + ((df * df) / (r * dx)) - (2.0 * (f_vals[i] + f_vals[i - 1]));

      if (r_val > local_max_r) {
        local_max_r = r_val;
        local_idx = static_cast<int>(i);
      }
    }

    struct MaxData {
      double value{};
      int index{};
    };

    MaxData local_data{local_max_r, local_idx};
    MaxData global_data{};

    MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (rank == 0) {
      const int idx = global_data.index;
      const double new_x = (0.5 * (x_vals[idx] + x_vals[idx - 1])) - ((f_vals[idx] - f_vals[idx - 1]) / (2.0 * m));

      x_vals.insert(x_vals.begin() + idx, new_x);
      f_vals.insert(f_vals.begin() + idx, f(new_x));
    }

    int n = static_cast<int>(x_vals.size());
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    x_vals.resize(static_cast<std::size_t>(n));
    f_vals.resize(static_cast<std::size_t>(n));

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
