#pragma once

// Только необходимые заголовки
#include <cstdint>  // NOLINT
#include <vector>   // NOLINT

// MPI заголовок
#include <mpi.h>

// Заголовок задачи
#include "task/include/task.hpp"  // NOLINT

namespace pikhotskiy_r_scatter {

struct ScatterArguments {
  const void *source_buffer{nullptr};
  int elements_to_send{0};
  MPI_Datatype send_data_type{MPI_DATATYPE_NULL};

  void *destination_buffer{nullptr};
  int elements_to_receive{0};
  MPI_Datatype receive_data_type{MPI_DATATYPE_NULL};

  int root_process{0};
  MPI_Comm communicator{MPI_COMM_NULL};
};

using InputType = ScatterArguments;
using OutputType = std::vector<std::uint8_t>;
using TestingType = ScatterArguments;
using TaskBase = ppc::task::Task<InputType, OutputType>;

}  // namespace pikhotskiy_r_scatter