#include "levonychev_i_mult_matrix_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "levonychev_i_mult_matrix_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_mult_matrix_vec {

LevonychevIMultMatrixVecMPI::LevonychevIMultMatrixVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
  GetOutput() = {};
}

bool LevonychevIMultMatrixVecMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const size_t matrix_size = std::get<0>(GetInput()).size();
    const int rows = std::get<1>(GetInput());
    const int cols = std::get<2>(GetInput());
    bool is_correct_matrix_size = (matrix_size == static_cast<size_t>(rows) * static_cast<size_t>(cols));
    bool is_correct_vector_size = (static_cast<size_t>(cols) == std::get<3>(GetInput()).size());
    return matrix_size != 0 && rows != 0 && cols != 0 && is_correct_matrix_size && is_correct_vector_size;
  }
  return true;
}

bool LevonychevIMultMatrixVecMPI::PreProcessingImpl() {
  int rank = 0;
  int rows = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    rows = std::get<1>(GetInput());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput().resize(rows);
  return true;
}

bool LevonychevIMultMatrixVecMPI::RunImpl() {
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  OutType &global_b = GetOutput();

  int rows = 0;
  int cols = 0;
  std::vector<int64_t> x;

  std::vector<int> recvcounts_scatterv(proc_num);
  std::vector<int> displs_scatterv(proc_num);
  std::vector<int> recvcounts_gatherv(proc_num);
  std::vector<int> displs_gatherv(proc_num);

  if (proc_rank == 0) {
    rows = std::get<1>(GetInput());
    cols = std::get<2>(GetInput());
    x = std::get<3>(GetInput());
    for (int i = 0; i < proc_num; ++i) {
      int local_count_of_rows = i == (proc_num - 1) ? ((rows / proc_num) + (rows % proc_num)) : (rows / proc_num);
      recvcounts_scatterv[i] = local_count_of_rows * cols;
      recvcounts_gatherv[i] = local_count_of_rows;
      int start = i * (rows / proc_num);
      displs_scatterv[i] = start * cols;
      displs_gatherv[i] = start;
    }
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  x.resize(cols);
  MPI_Bcast(x.data(), cols, MPI_INT64_T, 0, MPI_COMM_WORLD);

  int local_count_of_rows = (proc_rank == (proc_num - 1)) ? ((rows / proc_num) + (rows % proc_num)) : (rows / proc_num);
  int recvcount = local_count_of_rows * cols;
  OutType local_matrix(recvcount);
  MPI_Scatterv(std::get<0>(GetInput()).data(), recvcounts_scatterv.data(), displs_scatterv.data(), MPI_INT64_T,
               local_matrix.data(), recvcount, MPI_INT64_T, 0, MPI_COMM_WORLD);
  OutType local_b(local_count_of_rows);

  for (int i = 0; i < local_count_of_rows; ++i) {
    const int start = cols * i;
    int64_t scalar_product = 0;
    for (int j = 0; j < cols; ++j) {
      scalar_product += local_matrix[start + j] * x[j];
    }
    local_b[i] = scalar_product;
  }
  MPI_Gatherv(local_b.data(), local_count_of_rows, MPI_INT64_T, global_b.data(),
              recvcounts_gatherv.data(), displs_gatherv.data(), MPI_INT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_b.data(), rows, MPI_INT64_T, 0, MPI_COMM_WORLD);
  return true;
}

bool LevonychevIMultMatrixVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_mult_matrix_vec
