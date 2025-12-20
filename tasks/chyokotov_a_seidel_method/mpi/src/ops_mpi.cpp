#include "chyokotov_a_seidel_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "chyokotov_a_seidel_method/common/include/common.hpp"

namespace chyokotov_a_seidel_method {

ChyokotovASeidelMethodMPI::ChyokotovASeidelMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().first.clear();
  GetInput().first.reserve(in.first.size());
  for (const auto &row : in.first) {
    GetInput().first.push_back(row);
  }
  GetInput().second = in.second;
  GetOutput().clear();
}

bool ChyokotovASeidelMethodMPI::ValidationImpl() {
  int rank{};

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 1;

  if (rank == 0) {
    const auto &matrix = GetInput().first;
    const auto &vec = GetInput().second;

    if (!matrix.empty()) {
      if (vec.size() != matrix.size()) {
        is_valid = 0;
      }

      size_t n = matrix[0].size();
      for (const auto &row : matrix) {
        if (row.size() != n) {
          is_valid = 0;
          break;
        }
      }
      for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++) {
          if (i != j) {
            sum += std::abs(matrix[i][j]);
          }
        }
        if (std::abs(matrix[i][i]) <= sum) {
          is_valid = 0;
        }
      }
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return is_valid;
}

bool ChyokotovASeidelMethodMPI::PreProcessingImpl() {
  GetOutput().resize(GetInput().first.size(), 0.0);
  if (GetInput().first.empty()) {
    return true;
  }
  return true;
}

bool ChyokotovASeidelMethodMPI::RunImpl() {
  int rank{};
  int size{};

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n{};

  if (rank == 0) {
    auto &matrix = GetInput().first;
    n = static_cast<int>(matrix.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base = n / size;
  int rem = n % size;

  int local_rows = base + (rank < rem ? 1 : 0);
  int local_start = rank * base + std::min(rank, rem);

  std::vector<int> displs(size);
  std::vector<int> counts(size);

  for (int i = 0; i < size; i++) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }

  std::vector<std::vector<double>> my_A(local_rows, std::vector<double>(n));
  std::vector<double> my_b(local_rows);
  std::vector<double> my_x(local_rows);

  if (rank == 0) {
    const auto &matrix = GetInput().first;
    const auto &vector = GetInput().second;
    for (int i = 0; i < local_rows; i++) {
      my_A[i] = matrix[i];
      my_b[i] = vector[i];
    }

    for (int proc = 1; proc < size; proc++) {
      int proc_rows = base + (proc < rem ? 1 : 0);
      int proc_start = displs[proc];

      for (int i = 0; i < proc_rows; i++) {
        int row_idx = proc_start + i;
        MPI_Send(matrix[row_idx].data(), n, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
        MPI_Send(&vector[row_idx], 1, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
      }
    }
  } else {
    for (int i = 0; i < local_rows; i++) {
      MPI_Recv(my_A[i].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&my_b[i], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  std::vector<double> x(n, 0.0);

  for (int iter = 0; iter < 1000; iter++) {
    double local_max_diff = 0.0;

    for (int i = 0; i < local_rows; i++) {
      int global_i = local_start + i;
      double sum = my_b[i];

      for (int j = 0; j < global_i; j++) {
        sum -= my_A[i][j] * x[j];
      }
      for (int j = global_i + 1; j < n; j++) {
        sum -= my_A[i][j] * x[j];
      }

      double new_xi = sum / my_A[i][global_i];
      double diff = std::abs(new_xi - x[global_i]);

      if (diff > local_max_diff) {
        local_max_diff = diff;
      }

      my_x[i] = new_xi;
      x[global_i] = new_xi;
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    double global_diff{};
    MPI_Allreduce(&local_max_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (global_diff < 0.00001) {
      break;
    }
  }

  GetOutput() = x;

  return true;
}

bool ChyokotovASeidelMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_a_seidel_method
