#include "pikhotskiy_r_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace pikhotskiy_r_scatter {

PikhotskiyRScatterMPI::PikhotskiyRScatterMPI(const InputType &input_args) {
  SetTypeOfTask(GetTaskType());
  GetInput() = input_args;

  MPI_Comm_rank(input_args.communicator, &mpi_rank_);
  MPI_Comm_size(input_args.communicator, &mpi_size_);
  is_root_process_ = (mpi_rank_ == input_args.root_process);
}

bool PikhotskiyRScatterMPI::ValidationImpl() {
  const auto &params = GetInput();

  if (params.elements_to_send <= 0 || params.elements_to_send != params.elements_to_receive) {
    return false;
  }

  if (params.send_data_type != params.receive_data_type) {
    return false;
  }

  if (is_root_process_ && params.source_buffer == nullptr) {
    return false;
  }

  if (params.destination_buffer == nullptr) {
    return false;
  }

  auto validate_data_type = [](MPI_Datatype dt) { return dt == MPI_INT || dt == MPI_FLOAT || dt == MPI_DOUBLE; };

  return validate_data_type(params.send_data_type);
}

bool PikhotskiyRScatterMPI::PreProcessingImpl() {
  return true;
}

bool PikhotskiyRScatterMPI::RunImpl() {
  auto &parameters = GetInput();

  const void *send_ptr = is_root_process_ ? parameters.source_buffer : nullptr;
  void *receive_ptr = parameters.destination_buffer;

  int send_count = parameters.elements_to_send;
  MPI_Datatype data_type = parameters.send_data_type;
  int root = parameters.root_process;
  MPI_Comm comm = parameters.communicator;

  int mpi_result = MPI_Scatter(send_ptr, send_count, data_type, receive_ptr, send_count, data_type, root, comm);

  if (mpi_result != MPI_SUCCESS) {
    return false;
  }

  // Сохраняем результат
  int element_size;
  MPI_Type_size(data_type, &element_size);
  std::size_t total_bytes = static_cast<std::size_t>(send_count) * element_size;

  OutputType result(total_bytes);
  std::memcpy(result.data(), receive_ptr, total_bytes);

  GetOutput() = std::move(result);

  return true;
}

bool PikhotskiyRScatterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_scatter
