#include "gonozov_l_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "gonozov_l_elem_vec_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gonozov_l_elem_vec_sum {

GonozovLElemVecSumMPI::GonozovLElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GonozovLElemVecSumMPI::ValidationImpl() {
  return true;
}

bool GonozovLElemVecSumMPI::PreProcessingImpl() {
  vector_size_ = static_cast<int>(GetInput().size());
  return true;
  // int procRank = 0;
  //   MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

  //   if (procRank == 0) {
  //       vector_size_ = static_cast<int>(GetInput().size());
  //   }

  //   MPI_Bcast(&vector_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //   return true;
}

bool GonozovLElemVecSumMPI::RunImpl() {
  //   int procNum = 0;
  //   int procRank = 0;
  //   OutType result = 0;
  //   MPI_Comm_size(MPI_COMM_WORLD, &procNum);
  //   MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

  //   if (vector_size_ < procNum) {
  //     if (procRank == 0) {
  //       for (int i = 0; i < vector_size_; i++) {
  //         GetOutput() += GetInput()[i];
  //       }
  //     } else {
  //       GetOutput() = 0;
  //     }
  //     result = GetOutput();
  //     MPI_Bcast(&result, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  //     GetOutput() = result;
  //     return true;
  //   }

  //   int n = vector_size_ / procNum;
  //   int remainder = vector_size_ % procNum;
  //   int local_size = n + (procRank < remainder ? 1 : 0);

  //   int *subvector = new int[local_size];
  //   int *sendcounts = nullptr;
  //   int *displs = nullptr;

  //   if (procRank == 0) {
  //     sendcounts = new int[procNum];
  //     displs = new int[procNum];

  //     int offset = 0;
  //     for (int i = 0; i < procNum; i++) {
  //       sendcounts[i] = n + (i < remainder ? 1 : 0);
  //       displs[i] = offset;
  //       offset += sendcounts[i];
  //     }
  //   }

  //   MPI_Scatterv(procRank == 0 ? GetInput().data() : nullptr, procRank == 0 ? sendcounts : nullptr,
  //                procRank == 0 ? displs : nullptr, MPI_INT, subvector, local_size, MPI_INT, 0, MPI_COMM_WORLD);

  //   OutType local_sum = 0;
  //   for (int i = 0; i < local_size; i++) {
  //     local_sum += subvector[i];
  //   }

  //   OutType global_sum = 0;
  //   MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  //   GetOutput() = global_sum;

  //   delete[] subvector;
  //   if (procRank == 0) {
  //     delete[] sendcounts;
  //     delete[] displs;
  //   }

  //   return true;
  int procNum = 0;
  int procRank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &procNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

  if (vector_size_ < procNum) {
    OutType result = 0;
    if (procRank == 0) {
      for (int i = 0; i < vector_size_; i++) {
        result += GetInput()[i];
      }
    }
    MPI_Bcast(&result, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    GetOutput() = result;
    return true;
  }

  int n = vector_size_ / procNum;
  int remainder = vector_size_ % procNum;
  int local_size = n + (procRank < remainder ? 1 : 0);

  std::vector<int> sendcounts(procNum), displs(procNum);
  std::vector<int> subvector(local_size);

  if (procRank == 0) {
    int offset = 0;
    for (int i = 0; i < procNum; i++) {
      sendcounts[i] = n + (i < remainder ? 1 : 0);
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  MPI_Scatterv((procRank == 0) ? GetInput().data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
               subvector.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

  OutType local_sum = 0;
  for (int i = 0; i < local_size; i++) {
    local_sum += subvector[i];
  }

  OutType global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = global_sum;

  return true;
}

bool GonozovLElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gonozov_l_elem_vec_sum
