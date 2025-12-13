#include "telnov_strongin_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
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
  return in.eps > 0 && in.b > in.a;
}

bool TelnovStronginAlgorithmMPI::PreProcessingImpl() {
  return true;
}

bool TelnovStronginAlgorithmMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &in = GetInput();
  const double a = in.a;
  const double b = in.b;
  const double eps = in.eps;

  auto f = [](double x) { return (x - 1) * (x - 1) + 1; };

  std::vector<double> X = {a, b};
  std::vector<double> F = {f(a), f(b)};

  while ((X.back() - X.front()) > eps) {
    double M = 0.0;
    for (size_t i = 1; i < X.size(); ++i) {
      M = std::max(M, std::abs(F[i] - F[i - 1]) / (X[i] - X[i - 1]));
    }
    if (M == 0) {
      M = 1.0;
    }

    double r = 2.0;
    double local_maxR = -1e9;
    size_t local_t = 0;

    for (size_t i = rank + 1; i < X.size(); i += size) {
      double Ri = r * (X[i] - X[i - 1]) + (F[i] - F[i - 1]) * (F[i] - F[i - 1]) / (r * (X[i] - X[i - 1])) -
                  2 * (F[i] + F[i - 1]);
      if (Ri > local_maxR) {
        local_maxR = Ri;
        local_t = i;
      }
    }

    struct {
      double val;
      int idx;
    } local_data{local_maxR, (int)local_t}, global_data;
    MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (rank == 0) {
      double x_new =
          0.5 * (X[global_data.idx] + X[global_data.idx - 1]) - (F[global_data.idx] - F[global_data.idx - 1]) / (2 * M);
      X.insert(X.begin() + global_data.idx, x_new);
      F.insert(F.begin() + global_data.idx, f(x_new));
    }

    int n = X.size();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    X.resize(n);
    F.resize(n);
    MPI_Bcast(X.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(F.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  GetOutput() = *std::min_element(F.begin(), F.end());
  return true;
}

bool TelnovStronginAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace telnov_strongin_algorithm
