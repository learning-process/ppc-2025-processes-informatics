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

  // Валидацию делаем только на 0-м процессе, чтобы не дублировать логику
  if (rank == 0) {
    const auto &in_ = GetInput();
    if (in_.size() < 2) {
      return false;
    }
    rows = in_[0];
    cols = in_[1];
    if (rows <= 0 || cols <= 0 || static_cast<size_t>(rows * cols) != in_.size() - 2) {
      return false;
    }
  }
  return true;
}

bool MaxElementMatrMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Данные существуют и подготавливаются только на 0-м процессе
  if (rank == 0) {
    const auto &in_ = GetInput();
    rows = in_[0];
    cols = in_[1];
    matrix_.clear();
    matrix_.reserve(rows * cols);
    std::copy(in_.begin() + 2, in_.end(), std::back_inserter(matrix_));
  }

  // Рассылаем размеры матрицы всем процессам
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool MaxElementMatrMPI::RunImpl() {
  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Если матрица пустая, то и делать нечего
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

    // Раздаем части матрицы с 0-го процесса всем остальным (включая себя)
    MPI_Scatterv(matrix_.data(), sendcounts.data(), displs.data(), MPI_INT, recv_buf.data(), sendcounts[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    int local_max = (recv_buf.empty()) ? INT_MIN : recv_buf[0];
    for (int val : recv_buf) {
      if (val > local_max) {
        local_max = val;
      }
    }

    // Собираем локальные максимумы на 0-й процесс и вычисляем глобальный максимум
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  }

  // ====================== КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ======================
  // Рассылаем итоговый максимум с 0-го процесса на все остальные,
  // чтобы на этапе PostProcessing у всех был правильный ответ.
  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);
  // =================================================================

  return true;
}

bool MaxElementMatrMPI::PostProcessingImpl() {
  // Теперь КАЖДЫЙ процесс знает правильный ответ и может его записать.
  // Тестовый фреймворк проверит результат на всех процессах.
  GetOutput() = global_max;
  return true;
}

}  // namespace nikitina_v_max_elem_matr
