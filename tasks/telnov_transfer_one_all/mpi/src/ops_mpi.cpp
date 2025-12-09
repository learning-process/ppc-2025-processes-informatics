#include "telnov_transfer_one_all/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "telnov_transfer_one_all/common/include/common.hpp"
#include "util/include/util.hpp"

namespace telnov_transfer_one_all {

TelnovTransferOneAllMPI::TelnovTransferOneAllMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(0);
}

bool TelnovTransferOneAllMPI::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput().empty());
}

bool TelnovTransferOneAllMPI::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool TelnovTransferOneAllMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int count = static_cast<int>(GetInput().size());
  int root = root_;

  MPI_Datatype mpi_type;
  if constexpr (std::is_same<typename InType::value_type, int>::value) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same<typename InType::value_type, float>::value) {
    mpi_type = MPI_FLOAT;
  } else if constexpr (std::is_same<typename InType::value_type, double>::value) {
    mpi_type = MPI_DOUBLE;
  } else {
    return false;
  }

  auto data = GetInput();
  void *data_ptr = data.data();

  int virtual_rank = (rank + size - root) % size;
  int mask = 1;
  while (mask < size) {
    if ((virtual_rank & mask) == 0) {
      int dest = virtual_rank | mask;
      if (dest < size) {
        int real_dest = (dest + root) % size;
        MPI_Send(data_ptr, count, mpi_type, real_dest, 0, MPI_COMM_WORLD);
      }
    } else {
      int src = virtual_rank & (~mask);
      int real_src = (src + root) % size;
      MPI_Recv(data_ptr, count, mpi_type, real_src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      break;
    }
    mask <<= 1;
  }

  SetOutput(data);
  return true;
}

bool TelnovTransferOneAllMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace telnov_transfer_one_all
