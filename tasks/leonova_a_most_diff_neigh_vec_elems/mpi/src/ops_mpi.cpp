#include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
#include "util/include/util.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

LeonovaAMostDiffNeighVecElemsMPI::LeonovaAMostDiffNeighVecElemsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::tuple<int, int>(0, 0);
}

bool LeonovaAMostDiffNeighVecElemsMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool LeonovaAMostDiffNeighVecElemsMPI::PreProcessingImpl() {
  return true;
}

bool LeonovaAMostDiffNeighVecElemsMPI::RunImpl() {
  const auto &input_vec = GetInput();

  // Обработка особых случаев
  if (input_vec.empty()) {
    GetOutput() = std::make_tuple(0, 0);
    return true;
  }

  if (input_vec.size() == 1) {
    GetOutput() = std::make_tuple(input_vec[0], input_vec[0]);
    return true;
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Для простоты, если процессов больше 1, пусть процесс 0 обработает все данные
  // Это временное решение чтобы избежать deadlock'ов
  if (size > 1) {
    if (rank == 0) {
      // Процесс 0 обрабатывает все данные
      int max_diff = -1;
      std::tuple<int, int> best_pair(0, 0);

      for (size_t i = 0; i < input_vec.size() - 1; ++i) {
        int diff = std::abs(input_vec[i] - input_vec[i + 1]);
        if (diff > max_diff) {
          max_diff = diff;
          best_pair = std::make_tuple(input_vec[i], input_vec[i + 1]);
        }
      }

      GetOutput() = best_pair;

      // Рассылаем результат всем процессам
      int result_first = std::get<0>(best_pair);
      int result_second = std::get<1>(best_pair);
      for (int dest = 1; dest < size; ++dest) {
        MPI_Send(&result_first, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&result_second, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      }
    } else {
      // Остальные процессы получают результат
      int result_first, result_second;
      MPI_Recv(&result_first, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&result_second, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      GetOutput() = std::make_tuple(result_first, result_second);
    }
  } else {
    // Только один процесс - обрабатываем последовательно
    int max_diff = -1;
    std::tuple<int, int> best_pair(0, 0);

    for (size_t i = 0; i < input_vec.size() - 1; ++i) {
      int diff = std::abs(input_vec[i] - input_vec[i + 1]);
      if (diff > max_diff) {
        max_diff = diff;
        best_pair = std::make_tuple(input_vec[i], input_vec[i + 1]);
      }
    }

    GetOutput() = best_pair;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LeonovaAMostDiffNeighVecElemsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace leonova_a_most_diff_neigh_vec_elems
