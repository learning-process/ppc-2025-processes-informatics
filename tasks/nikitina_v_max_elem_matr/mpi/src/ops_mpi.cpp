#include "nikitina_v_max_elem_matr/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <vector>

namespace nikitina_v_max_elem_matr {

MaxElementMatrMPI::MaxElementMatrMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaxElementMatrMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Используем int для результата, т.к. MPI_Bcast с bool может быть капризным
  int validation_result = 1;  // 1 = true, 0 = false

  if (rank == 0) {
    const auto &in_ = GetInput();
    if (in_.size() < 2) {
      validation_result = 0;
    } else {
      rows = in_[0];
      cols = in_[1];
      if (rows < 0 || cols < 0 || static_cast<size_t>(rows * cols) != in_.size() - 2) {
        validation_result = 0;
      }
    }
  }

  // ================== КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ==================
  // Процесс 0 рассылает результат валидации всем остальным процессам.
  MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
  // =======================================================

  return validation_result == 1;
}

bool MaxElementMatrMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &in_ = GetInput();
    rows = in_[0];
    cols = in_[1];
    if (rows > 0 && cols > 0) {
      matrix_.clear();
      matrix_.reserve(rows * cols);
      std::copy(in_.begin() + 2, in_.end(), std::back_inserter(matrix_));
    }
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool MaxElementMatrMPI::RunImpl() {
  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rows * cols == 0) {
    global_max = INT_MIN;
  } else {
    const int total_elements = rows * cols;
    const int elements_per_proc = total_elements / world_size;
    const int remainder_elements = total_elements % world_size;

    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);
    int current_displ = 0;

    for (int i = 0; i < world_size; ++i) {
      sendcounts[i] = (i < remainder_elements) ? elements_per_proc + 1 : elements_per_proc;
      displs[i] = current_displ;
      current_displ += sendcounts[i];
    }

    std::vector<int> recv_buf(sendcounts[rank]);

    MPI_Scatterv(matrix_.data(), sendcounts.data(), displs.data(), MPI_INT, recv_buf.data(), sendcounts[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    int local_max = (recv_buf.empty()) ? INT_MIN : recv_buf[0];
    for (int val : recv_buf) {
      if (val > local_max) {
        local_max = val;
      }
    }

    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  }

  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool MaxElementMatrMPI::PostProcessingImpl() {
  GetOutput() = global_max;
  return true;
}

}  // namespace nikitina_v_max_elem_matr
