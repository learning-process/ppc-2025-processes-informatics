#include "savva_d_min_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "savva_d_min_elem_vec/common/include/common.hpp"

namespace savva_d_min_elem_vec {

SavvaDMinElemVecMPI::SavvaDMinElemVecMPI(const InType &in) {  // эта функция не изменяется в задачах
  SetTypeOfTask(GetStaticTypeOfTask());                       // конструктор правильная постановка задачи
  GetInput() = in;  // GetInput() нужен чтобы больше не использовать сами данные in
  GetOutput() = 0;
}

bool SavvaDMinElemVecMPI::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool SavvaDMinElemVecMPI::PreProcessingImpl() {
  return true;
}

bool SavvaDMinElemVecMPI::RunImpl() {
  auto &input_vec = GetInput();
  if (input_vec.empty()) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = static_cast<int>(input_vec.size());
  int elements_per_proc = n / size;
  int remainder = n % size;

  int local_start = (rank * elements_per_proc) + std::min(rank, remainder);
  int local_end = local_start + elements_per_proc + (rank < remainder ? 1 : 0);

  int local_min = std::numeric_limits<int>::max();
  for (int i = local_start; i < local_end && i < n; ++i) {
    local_min = std::min(input_vec[i], local_min);
  }

  int global_min = 0;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_min;

  // Синхронизация
  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool SavvaDMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace savva_d_min_elem_vec
