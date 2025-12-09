#include "telnov_transfer_one_all/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <type_traits>

namespace telnov_transfer_one_all {

template <typename T>
TelnovTransferOneAllMPI<T>::TelnovTransferOneAllMPI(const InType &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
  this->GetOutput().resize(0);
}

template <typename T>
bool TelnovTransferOneAllMPI<T>::ValidationImpl() {
  return (!this->GetInput().empty()) && (this->GetOutput().empty());
}

template <typename T>
bool TelnovTransferOneAllMPI<T>::PreProcessingImpl() {
  return true;
}

template <typename T>
bool TelnovTransferOneAllMPI<T>::RunImpl() {
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);

  if (is_mpi_initialized == 0) {
    this->GetOutput() = this->GetInput();
    return true;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int count = static_cast<int>(this->GetInput().size());
  int root = 0;

  MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
  if constexpr (std::is_same_v<T, int>) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same_v<T, float>) {
    mpi_type = MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    mpi_type = MPI_DOUBLE;
  } else {
    return false;
  }

  auto data = this->GetInput();
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

  this->GetOutput() = data;
  return true;
}

template <typename T>
bool TelnovTransferOneAllMPI<T>::PostProcessingImpl() {
  return !this->GetOutput().empty();
}

template class TelnovTransferOneAllMPI<int>;
template class TelnovTransferOneAllMPI<float>;
template class TelnovTransferOneAllMPI<double>;

}  // namespace telnov_transfer_one_all
