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

  if (!ValidationImpl()) {
    return false;
  }

  if (input_vec.size() == 1) {
    GetOutput() = std::make_tuple(input_vec[0], input_vec[0]);
    return true;
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Для простоты, если процессов больше 1, используем упрощенный подход
  // чтобы избежать deadlock'ов с граничными элементами
  if (size > 1) {
    // Рассылаем все данные всем процессам через MPI_Bcast
    int total_size = static_cast<int>(input_vec.size());
    MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> all_data(total_size);
    if (rank == 0) {
      all_data = input_vec;
    }
    MPI_Bcast(all_data.data(), total_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Каждый процесс обрабатывает свою часть
    int local_size = total_size / size;
    int remainder = total_size % size;
    
    int start = rank * local_size + std::min(rank, remainder);
    int end = start + local_size + (rank < remainder ? 1 : 0);
    
    // Убедимся что end не превышает total_size - 1
    if (end > total_size - 1) {
      end = total_size - 1;
    }

    // Локальный поиск в своей части
    int local_max_diff = -1;
    int local_first = 0;
    int local_second = 0;

    for (int i = start; i < end; ++i) {
      int diff = std::abs(all_data[i] - all_data[i + 1]);
      if (diff > local_max_diff) {
        local_max_diff = diff;
        local_first = all_data[i];
        local_second = all_data[i + 1];
      }
    }

    // Собираем результаты
    std::vector<int> all_diffs(size);
    std::vector<int> all_firsts(size);
    std::vector<int> all_seconds(size);

    MPI_Gather(&local_max_diff, 1, MPI_INT, all_diffs.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_first, 1, MPI_INT, all_firsts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_second, 1, MPI_INT, all_seconds.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Процесс 0 находит глобальный максимум
    if (rank == 0) {
      int global_max_diff = -1;
      int global_first = 0;
      int global_second = 0;

      for (int i = 0; i < size; ++i) {
        if (all_diffs[i] > global_max_diff) {
          global_max_diff = all_diffs[i];
          global_first = all_firsts[i];
          global_second = all_seconds[i];
        }
      }
      GetOutput() = std::make_tuple(global_first, global_second);
    }

    // Рассылаем результат
    int result_first = std::get<0>(GetOutput());
    int result_second = std::get<1>(GetOutput());
    MPI_Bcast(&result_first, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_second, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
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