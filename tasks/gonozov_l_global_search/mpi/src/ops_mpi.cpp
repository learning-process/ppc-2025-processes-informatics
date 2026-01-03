#include "gonozov_l_global_search/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include "gonozov_l_global_search/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gonozov_l_global_search {

GonozovLGlobalSearchMPI::GonozovLGlobalSearchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool GonozovLGlobalSearchMPI::ValidationImpl() {
  return (std::get<1>(GetInput()) > 1.0) && (std::get<2>(GetInput()) < std::get<3>(GetInput())) &&
         (std::get<4>(GetInput()) > 0);
}

bool GonozovLGlobalSearchMPI::PreProcessingImpl() {
  return true;
}

namespace {
double Countingm(double M, double r) {
  return (M == 0.0) ? 1.0 : r * M;
}

double CountMIncremental(int t, double M, const std::vector<double> &x, const std::function<double(double)> &f) {
  if (M == -std::numeric_limits<double>::infinity()) {
    return std::abs((f(x[1]) - f(x[0])) / (x[1] - x[0]));
  }

  double M1 = std::abs((f(x.back()) - f(x[t - 1])) / (x.back() - x[t - 1]));
  double M2 = std::abs((f(x[t]) - f(x.back())) / (x[t] - x.back()));

  return std::max(M, std::max(M1, M2));
}

double IntervalCharacteristic(double x1, double x2, double f1, double f2, double m) {
  double dx = x2 - x1;
  return m * dx + (f2 - f1) * (f2 - f1) / (m * dx) - 2.0 * (f1 + f2);
}

}  // namespace
bool GonozovLGlobalSearchMPI::RunImpl() {
  auto function = std::get<0>(GetInput());
  double r = std::get<1>(GetInput());
  double a = std::get<2>(GetInput());
  double b = std::get<3>(GetInput());
  double eps = std::get<4>(GetInput());

  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  std::vector<double> testSequence;
  double global_min_x = a;
  double global_min_value = function(a);

  int t = 1;
  double M = -std::numeric_limits<double>::infinity();

  if (proc_rank == 0) {
    testSequence = {a, b};
    if (function(b) < global_min_value) {
      global_min_x = b;
      global_min_value = function(b);
    }
  }

  bool continue_iteration = true;

  while (continue_iteration) {
    int n = 0;
    if (proc_rank == 0) {
      n = testSequence.size();
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (proc_rank != 0) {
      testSequence.resize(n);
    }

    MPI_Bcast(testSequence.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::sort(testSequence.begin(), testSequence.end());

    double m = 0.0;
    if (proc_rank == 0) {
      M = CountMIncremental(t, M, testSequence, function);
      m = Countingm(M, r);
    }
    MPI_Bcast(&m, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int intervals = n - 1;
    int per_proc = intervals / proc_num;
    int rem = intervals % proc_num;

    int l = proc_rank * per_proc + std::min(proc_rank, rem);
    int r_i = l + per_proc + (proc_rank < rem ? 1 : 0);

    double local_max = -std::numeric_limits<double>::infinity();
    int local_idx = -1;

    if (l < r_i) {
      for (int i = l + 1; i <= r_i; ++i) {
        double R = IntervalCharacteristic(testSequence[i - 1], testSequence[i], function(testSequence[i - 1]),
                                          function(testSequence[i]), m);
        if (R > local_max) {
          local_max = R;
          local_idx = i;
        }
      }
    }

    struct {
      double val;
      int idx;
    } loc, glob;
    loc.val = local_max;
    loc.idx = local_idx;

    MPI_Reduce(&loc, &glob, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    if (proc_rank == 0) {
      if (glob.idx <= 0 || glob.idx >= static_cast<int>(testSequence.size())) {
        continue_iteration = false;
      } else {
        t = glob.idx;
        if (t < 1) {
          break;
        }

        double x_new = 0.5 * (testSequence[t] + testSequence[t - 1]) -
                       (function(testSequence[t]) - function(testSequence[t - 1])) / (2.0 * m);

        double fx = function(x_new);
        if (fx < global_min_value) {
          global_min_value = fx;
          global_min_x = x_new;
        }
        testSequence.push_back(x_new);
      }

      continue_iteration = std::abs(testSequence[t] - testSequence[t - 1]) > eps;
    }

    MPI_Bcast(&continue_iteration, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  }

  // рассылаем результат всем процессам
  MPI_Bcast(&global_min_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = global_min_x;

  return true;
}

bool GonozovLGlobalSearchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gonozov_l_global_search
