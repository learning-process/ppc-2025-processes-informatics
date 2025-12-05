#include "smyshlaev_a_mat_mul/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "smyshlaev_a_mat_mul/common/include/common.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_mat_mul {

namespace {

  void CalculateDistribution(int total_len, int proc_count, std::vector<int> &counts, std::vector<int> &offsets) {
  const int chunk = total_len / proc_count;
  const int remainder = total_len % proc_count;
  int offset = 0;
  for (int i = 0; i < proc_count; i++) {
    counts[i] = chunk + (i < remainder ? 1 : 0);
    offsets[i] = offset;
    offset += counts[i];
  }
}
}  // namespace

SmyshlaevAMatMulMPI::SmyshlaevAMatMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }

}

bool SmyshlaevAMatMulMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto& num_rows_a = std::get<0>(GetInput());
    const auto& mat_a = std::get<1>(GetInput());
    const auto& num_rows_b = std::get<2>(GetInput());
    const auto& mat_b = std::get<3>(GetInput());

    if (num_rows_a <= 0 || num_rows_b <= 0) {
      return false;
    }

    if (mat_a.empty() || mat_b.empty()) {
      return false;
    }

    if (mat_a.size() % num_rows_a != 0 || mat_b.size() % num_rows_b != 0) {
      return false;
    }

    const auto& num_cols_a = static_cast<int>(mat_a.size()) / num_rows_a;
    return (num_cols_a == num_rows_b);
  }

  return true;
}

bool SmyshlaevAMatMulMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto& num_rows_b = std::get<2>(GetInput());
    const auto& mat_b = std::get<3>(GetInput());
    const auto num_cols_b = static_cast<int>(mat_b.size()) / num_rows_b;
    
    mat_b_transposed.resize(mat_b.size());

    for (int i = 0; i < num_rows_b; ++i) {
      for (int j = 0; j < num_cols_b; ++j) {
        mat_b_transposed[j * num_rows_b + i] = mat_b[i * num_cols_b + j];
      }
    }
  }
  return true;
}

bool SmyshlaevAMatMulMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int num_rows_a = 0;
  int num_cols_a = 0;
  int num_cols_b = 0;

  if (rank == 0) {
    num_rows_a = std::get<0>(GetInput());
    const auto& mat_a = std::get<1>(GetInput());
    num_cols_a = static_cast<int>(mat_a.size()) / num_rows_a;
    const auto& num_rows_b = std::get<2>(GetInput());
    const auto& mat_b = std::get<3>(GetInput());
    num_cols_b = static_cast<int>(mat_b.size()) / num_rows_b;
  }

  MPI_Bcast(&num_rows_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&num_cols_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&num_cols_b, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts_a(size);
  std::vector<int> offsets_a(size);
  std::vector<int> sendcounts_b(size);
  std::vector<int> offsets_b(size);

  CalculateDistribution(num_rows_a, size, sendcounts_a, offsets_a);

  for (int i = 0; i < size; ++i) {
    sendcounts_a[i] *= num_cols_a;
    offsets_a[i] *= num_cols_a;
  }

  CalculateDistribution(num_cols_b, size, sendcounts_b, offsets_b);

  for (int i = 0; i < size; ++i) {
    sendcounts_b[i] *= num_cols_a;
    offsets_b[i] *= num_cols_a;
  }

  int my_rows_a = sendcounts_a[rank] / num_cols_a;

  std::vector<double> local_a(sendcounts_a[rank]);
  std::vector<double> local_b(sendcounts_b[rank]);

  int max_elem_count_b = 0;
  for(int c : sendcounts_b) max_elem_count_b = std::max(max_elem_count_b, c);
  std::vector<double> local_b_next(max_elem_count_b);

  const double* sendbuf_a = (rank == 0) ? std::get<1>(GetInput()).data() : nullptr;
  MPI_Scatterv(sendbuf_a, sendcounts_a.data(), offsets_a.data(), MPI_DOUBLE,
               local_a.data(), sendcounts_a[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  const double* sendbuf_b = (rank == 0) ? mat_b_transposed_.data() : nullptr;
  MPI_Scatterv(sendbuf_b, sendcounts_b.data(), offsets_a.data(), MPI_DOUBLE,
               local_b.data(), sendcounts_b[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_c(my_rows_a * num_cols_b, 0.0);
  
  int left_neighbor = (rank - 1 + size) % size;
  int right_neighbor = (rank + 1) % size;

  for (int step = 0; step < size; ++step) {
    int b_owner_rank = (rank - step + size) % size;
    int current_cols_b_count = sendcounts_b[b_owner_rank] / num_cols_a;
    int global_col_shift = displs_b[b_owner_rank] / num_cols_a;
    for (int i = 0; i < my_rows_a; ++i) {
      for (int j = 0; j < current_cols_b_count; ++j) {
        double sum = 0.0;
        for (int k = 0; k < num_cols_a; ++k) {
          sum += local_a[i * num_cols_a + k] * local_b[j * num_cols_a + k];
        }
        local_c[i * num_cols_b + (global_col_shift + j)] = sum;
      }
    }

    int send_count = sendcounts_b[b_owner_rank];
    int recv_owner_rank = (rank - (step + 1) + size) % size;
    int recv_count = sendcounts_b[recv_owner_rank];

    MPI_Sendrecv(local_b.data(), send_count, MPI_DOUBLE, right_neighbor, 0,
                 local_b_next.data(), recv_count, MPI_DOUBLE, left_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::swap(local_b, local_b_next);
  }

  std::vector<double> final_res;
  if (rank == 0) {
    final_res.resize(num_rows_a * num_cols_b);
  }

  std::vector<int> recvcounts_c(size);
  std::vector<int> displs_c(size);
  
  if (rank == 0) {
    int offset = 0;
    for(int i=0; i<size; ++i) {
        int r_rows = sendcounts_a[i] / num_cols_a;
        recvcounts_c[i] = r_rows * num_cols_b;
        displs_c[i] = offset;
        offset += recvcounts_c[i];
    }
  }

  MPI_Gatherv(local_c.data(), static_cast<int>(local_c.size()), MPI_DOUBLE,
              final_res.data(), recvcounts_c.data(), displs_c.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = final_res;
  }
  return true;
}

bool SmyshlaevAMatMulMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_mat_mul
