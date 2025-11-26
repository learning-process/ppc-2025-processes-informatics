#include "ashihmin_d_sum_of_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include "ashihmin_d_sum_of_elem/common/include/common.hpp"

namespace ashihmin_d_sum_of_elem {

AshihminDElemVecsSumMPI::AshihminDElemVecsSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AshihminDElemVecsSumMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool AshihminDElemVecsSumMPI::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

// bool AshihminDElemVecsSumMPI::RunImpl() { //решение
//   auto input = GetInput();
//   if (input == 0) {
//     return false;
//   }

//   for (InType i = 0; i < GetInput(); i++) {
//     for (InType j = 0; j < GetInput(); j++) {
//       for (InType k = 0; k < GetInput(); k++) {
//         std::vector<InType> tmp(i + j + k, 1);
//         GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
//         GetOutput() -= i + j + k;
//       }
//     }
//   }

//   const int num_threads = ppc::util::GetNumThreads();
//   GetOutput() *= num_threads;

//   int rank = 0;
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//   if (rank == 0) {
//     GetOutput() /= num_threads;
//   } else {
//     int counter = 0;
//     for (int i = 0; i < num_threads; i++) {
//       counter++;
//     }

//     if (counter != 0) {
//       GetOutput() /= counter;
//     }
//   }

//   MPI_Barrier(MPI_COMM_WORLD);
//   return GetOutput() > 0;
// }

// bool AshihminDElemVecsSumMPI::PostProcessingImpl() {
//   GetOutput() -= GetInput();
//   return GetOutput() > 0;
// }

bool AshihminDElemVecsSumMPI::RunImpl() {
  int n = GetInput();
  if (n <= 0) {
    return false;
  }

  int size = 0 int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int chunk = n / size;
  int start = rank * chunk;
  int end = (rank == size - 1 ? n : start + chunk);

  int local_sum = 0;
  for (int i = start; i < end; i++) {
    local_sum += 1;
  }

  int global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_sum;

  return true;
}

bool AshihminDElemVecsSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ashihmin_d_sum_of_elem
