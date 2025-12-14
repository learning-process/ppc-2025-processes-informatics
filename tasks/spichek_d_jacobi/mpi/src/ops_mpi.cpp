#include "spichek_d_jacobi/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "spichek_d_jacobi/common/include/common.hpp"

namespace spichek_d_jacobi {

SpichekDJacobiMPI::SpichekDJacobiMPI(const InType &in) : BaseTask(in) {  // <-- Используем список инициализации
  SetTypeOfTask(GetStaticTypeOfTask());
  // GetInput() = in; // УДАЛИТЬ или ЗАКОММЕНТИРОВАТЬ
  GetOutput() = Vector{};
}

bool SpichekDJacobiMPI::ValidationImpl() {
  const auto &[A, b, eps, max_iter] = GetInput();
  size_t n = A.size();

  if (n == 0) {
    return true;
  }
  if (A[0].size() != n || b.size() != n) {
    return false;
  }

  for (size_t i = 0; i < n; ++i) {
    if (std::abs(A[i][i]) < 1e-12) {
      return false;
    }
  }

  return true;
}

bool SpichekDJacobiMPI::PreProcessingImpl() {
  return true;
}

bool SpichekDJacobiMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[A_global, b_global, eps, max_iter] = GetInput();

  // ---------- 1. Размер задачи ----------
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(A_global.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    GetOutput() = Vector{};
    return true;
  }

  // ---------- 2. Разбиение ----------
  std::vector<int> counts(size), displs(size);

  int base = n / size;
  int rem = n % size;

  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }

  int local_n = counts[rank];

  // ---------- 3. Локальные данные ----------
  std::vector<double> local_A(local_n * n);
  Vector local_b(local_n);

  Vector x(n, 0.0);
  Vector x_new(n, 0.0);

  // ---------- 4. Рассылка ----------
  if (rank == 0) {
    for (int p = 0; p < size; ++p) {
      for (int i = 0; i < counts[p]; ++i) {
        int gi = displs[p] + i;

        if (p == 0) {
          std::copy(A_global[gi].begin(), A_global[gi].end(), local_A.begin() + i * n);
          local_b[i] = b_global[gi];
        } else {
          MPI_Send(A_global[gi].data(), n, MPI_DOUBLE, p, gi, MPI_COMM_WORLD);
          MPI_Send(&b_global[gi], 1, MPI_DOUBLE, p, gi + n, MPI_COMM_WORLD);
        }
      }
    }
  } else {
    for (int i = 0; i < local_n; ++i) {
      int gi = displs[rank] + i;
      MPI_Status st;
      MPI_Recv(local_A.data() + i * n, n, MPI_DOUBLE, 0, gi, MPI_COMM_WORLD, &st);
      MPI_Recv(&local_b[i], 1, MPI_DOUBLE, 0, gi + n, MPI_COMM_WORLD, &st);
    }
  }

  // ---------- 5. Итерации Якоби ----------
  double max_diff = 0.0;
  int iter = 0;

  do {
    ++iter;
    std::vector<double> local_x_new(local_n);

    for (int i = 0; i < local_n; ++i) {
      int gi = displs[rank] + i;
      const double *row = local_A.data() + i * n;

      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (j != gi) {
          sum += row[j] * x[j];
        }
      }

      local_x_new[i] = (local_b[i] - sum) / row[gi];
    }

    std::fill(x_new.begin(), x_new.end(), 0.0);

    MPI_Allgatherv(local_x_new.data(), local_n, MPI_DOUBLE, x_new.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    double local_max = 0.0;
    for (int i = 0; i < local_n; ++i) {
      int gi = displs[rank] + i;
      local_max = std::max(local_max, std::abs(x_new[gi] - x[gi]));
    }

    MPI_Allreduce(&local_max, &max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    x.swap(x_new);

  } while (max_diff > eps && iter < max_iter);

  // ---------- 6. Результат ----------
  if (rank == 0) {
    GetOutput() = x;  // БЕЗ if(rank == 0)
    return true;
  }
  GetOutput() = Vector{};
  return true;
}

bool SpichekDJacobiMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_jacobi
