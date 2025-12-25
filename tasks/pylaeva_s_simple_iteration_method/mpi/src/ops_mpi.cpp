#include "pylaeva_s_simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"

namespace pylaeva_s_simple_iteration_method {

PylaevaSSimpleIterationMethodMPI::PylaevaSSimpleIterationMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool PylaevaSSimpleIterationMethodMPI::ValidationImpl() {
  const auto &n = std::get<0>(GetInput());
  const auto &A = std::get<1>(GetInput());
  const auto &b = std::get<2>(GetInput());
  return ((n > 0) && (A.size() == n * n) && (b.size() == n) && (NotNullDeterm(A, n)) && (DiagonalDominance(A, n)));
}

bool PylaevaSSimpleIterationMethodMPI::PreProcessingImpl() {
  return true;
}

bool PylaevaSSimpleIterationMethodMPI::RunImpl() {
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  size_t n = 0;
  std::vector<double> A;
  std::vector<double> b;

  if (proc_rank == 0) {
    n = std::get<0>(GetInput());
    A = std::get<1>(GetInput());
    b = std::get<2>(GetInput());
  }

  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  std::vector<int> row_counts_per_rank(proc_num);
  std::vector<int> row_offsets_per_rank(proc_num);
  CalculateRowsDistribution(proc_num, static_cast<int>(n), row_counts_per_rank, row_offsets_per_rank);

  std::vector<int> mat_block_sizes(proc_num, 0);
  std::vector<int> mat_block_offsets(proc_num, 0);

  size_t base = n / proc_num;
  size_t rem = n % proc_num;

  for (int i = 0; i < proc_num; ++i) {
    int rows = base + (i < rem ? 1 : 0);
    mat_block_sizes[i] = rows * n;
    if (i > 0) {
      mat_block_offsets[i] = mat_block_offsets[i - 1] + mat_block_sizes[i - 1];
    }
  }

  int local_rows = row_counts_per_rank[proc_rank];
  int local_matrix_size = mat_block_sizes[proc_rank];

  std::vector<double> local_a(local_matrix_size);
  std::vector<double> local_b(local_rows);

  MPI_Scatterv(proc_rank == 0 ? A.data() : nullptr, mat_block_sizes.data(), mat_block_offsets.data(), MPI_DOUBLE,
               local_a.data(), local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(proc_rank == 0 ? b.data() : nullptr, row_counts_per_rank.data(), row_offsets_per_rank.data(), MPI_DOUBLE,
               local_b.data(), local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int start = row_offsets_per_rank[proc_rank];
  int count = local_rows;

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(count, 0.0);

  std::vector<int> recv_counts(proc_num);
  std::vector<int> allgather_displs(proc_num);
  CalculateRowsDistribution(proc_num, static_cast<int>(n), recv_counts, allgather_displs);

  for (int iter = 0; iter < kMaxIterations; ++iter) {
    for (int i = 0; i < count; ++i) {
      int global_i = start + i;
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (std::cmp_not_equal(j, global_i)) {
          sum += local_a[(i * n) + j] * x[j];
        }
      }
      local_x_new[i] = (local_b[i] - sum) / local_a[(i * n) + global_i];
    }

    MPI_Allgatherv(local_x_new.data(), count, MPI_DOUBLE, x_new.data(), recv_counts.data(), allgather_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double local_norm = 0.0;
    for (int i = 0; i < count; ++i) {
      int gi = start + i;
      double diff = x_new[gi] - x[gi];
      local_norm += diff * diff;
    }
    double global_norm = 0.0;
    MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    x = x_new;

    if (std::sqrt(global_norm) < kEps) {
      break;
    }
  }

  GetOutput() = x;

  return true;
}

bool PylaevaSSimpleIterationMethodMPI::PostProcessingImpl() {
  return true;
}

bool PylaevaSSimpleIterationMethodMPI::NotNullDeterm(const std::vector<double> &a, size_t n) {
  std::vector<double> tmp = a;

  for (size_t i = 0; i < n; i++) {
    // Поиск строки с ненулевым элементом в i-м столбце
    if (std::fabs(tmp[(i * n) + i]) < 1e-10) {
      // Текущий диагональный элемент близок к нулю
      // Ищем строку ниже с ненулевым элементом в этом столбце
      bool found = false;
      for (size_t j = i + 1; j < n; j++) {
        if (std::fabs(tmp[(j * n) + i]) > 1e-10) {
          // Меняем строки местами
          for (size_t k = i; k < n; k++) {
            std::swap(tmp[(i * n) + k], tmp[(j * n) + k]);
          }
          found = true;
          break;
        }
      }
      // Если не нашли подходящую строку для замены, определитель = 0
      if (!found) {
        return false;
      }
    }

    // Зануляем элементы ниже диагонали
    double pivot = tmp[(i * n) + i];
    for (size_t j = i + 1; j < n; j++) {
      double factor = tmp[(j * n) + i] / pivot;
      for (size_t k = i; k < n; k++) {
        tmp[(j * n) + k] -= tmp[(i * n) + k] * factor;
      }
    }
  }

  // Проверяем, что все диагональные элементы не равны нулю
  for (size_t i = 0; i < n; i++) {
    if (std::fabs(tmp[(i * n) + i]) < 1e-10) {
      return false;
    }
  }

  return true;
}

bool PylaevaSSimpleIterationMethodMPI::DiagonalDominance(const std::vector<double> &a, size_t n) {
  for (size_t i = 0; i < n; i++) {
    double diag = std::fabs(a[(i * n) + i]);  // Модуль диагонального элемента
    double row_sum = 0.0;                     // Сумма модулей недиагональных элементов строки

    for (size_t j = 0; j < n; j++) {
      if (j != i) {
        row_sum += std::fabs(a[(i * n) + j]);
      }
    }
    // Проверка строгого диагонального преобладания:
    // Диагональный элемент должен быть БОЛЬШЕ суммы остальных элементов строки
    if (diag <= row_sum) {
      return false;
    }
  }
  return true;
}

void PylaevaSSimpleIterationMethodMPI::CalculateRowsDistribution(int proc_num, int n,
                                                                 std::vector<int> &row_counts_per_rank,
                                                                 std::vector<int> &row_offsets_per_rank) {
  if (row_offsets_per_rank.empty()) {
    return;
  }
  size_t base = n / proc_num;
  size_t rem = n % proc_num;

  row_offsets_per_rank[0] = 0;
  for (int i = 0; i < proc_num; ++i) {
    row_counts_per_rank[i] = base + (i < rem ? 1 : 0);
    if (i > 0) {
      row_offsets_per_rank[i] = row_offsets_per_rank[i - 1] + row_counts_per_rank[i - 1];
    }
  }
}

}  // namespace pylaeva_s_simple_iteration_method
