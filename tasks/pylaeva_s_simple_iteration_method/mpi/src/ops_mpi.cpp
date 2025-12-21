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

  const auto &n = std::get<0>(GetInput());
  const auto &A = std::get<1>(GetInput());
  const auto &b = std::get<2>(GetInput());

  int base = static_cast<int>(n) / proc_num;
  int rem = static_cast<int>(n) % proc_num;
  int start = (proc_rank * base) + std::min(proc_rank, rem);
  int count = base + (proc_rank < rem ? 1 : 0);

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(count, 0.0);

  std::vector<int> recv_counts(proc_num, 0);
  std::vector<int> displs(proc_num, 0);
  recv_counts[0] = base + (0 < rem ? 1 : 0);
  for (int i = 1; i < proc_num; ++i) {
    recv_counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = displs[i - 1] + recv_counts[i - 1];
  }

  for (int iter = 0; iter < MaxIterations; ++iter) {
    for (int i = 0; i < count; ++i) {
      int global_i = start + i;
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (std::cmp_not_equal(j, global_i)) {
          sum += A[(global_i * n) + j] * x[j];
        }
      }
      local_x_new[i] = (b[global_i] - sum) / A[(global_i * n) + global_i];
    }

    MPI_Allgatherv(local_x_new.data(), count, MPI_DOUBLE, x_new.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    double local_norm = 0.0;
    for (int i = 0; i < count; ++i) {
      int gi = start + i;
      double diff = x_new[gi] - x[gi];
      local_norm += diff * diff;
    }

    double global_norm = 0.0;
    MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    x = x_new;
    if (std::sqrt(global_norm) < EPS) {
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

}  // namespace pylaeva_s_simple_iteration_method
