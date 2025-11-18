#include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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

  int total_size = static_cast<int>(input_vec.size());

  if (size > 1) {
    // распределяем данные по кусочкам
    int chunk_size = total_size / size;
    int remainder = total_size % size;

    // получаем на 1 элемент больше для проверки соседних пар на границах чанков
    int my_size = chunk_size + (rank < remainder ? 1 : 0) + 1;
    int my_offset = rank * chunk_size + std::min(rank, remainder);

    if (rank == size - 1 && my_offset + my_size > total_size) {
      my_size = total_size - my_offset;
    }

    std::vector<int> local_data(my_size);

    if (rank == 0) {
      for (int i = 0; i < my_size; ++i) {
        local_data[i] = input_vec[i];
      }
      // рассылаем
      for (int dest = 1; dest < size; ++dest) {
        int dest_size = chunk_size + (dest < remainder ? 1 : 0) + 1;
        int dest_offset = dest * chunk_size + std::min(dest, remainder);

        if (dest == size - 1 && dest_offset + dest_size > total_size) {
          dest_size = total_size - dest_offset;
        }

        MPI_Send(input_vec.data() + dest_offset, dest_size, MPI_INT, dest, 0, MPI_COMM_WORLD);
      }
    } else {
      MPI_Recv(local_data.data(), my_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int local_max_diff = -1;
    int local_first = 0;
    int local_second = 0;

    for (int i = 0; i < my_size - 1; ++i) {
      int diff = std::abs(local_data[i] - local_data[i + 1]);
      if (diff > local_max_diff) {
        local_max_diff = diff;
        local_first = local_data[i];
        local_second = local_data[i + 1];
      }
    }

    std::vector<int> all_diffs(size);
    std::vector<int> all_firsts(size);
    std::vector<int> all_seconds(size);

    MPI_Gather(&local_max_diff, 1, MPI_INT, all_diffs.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_first, 1, MPI_INT, all_firsts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_second, 1, MPI_INT, all_seconds.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

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

    // результат
    int result_first = std::get<0>(GetOutput());
    int result_second = std::get<1>(GetOutput());
    MPI_Bcast(&result_first, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_second, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      GetOutput() = std::make_tuple(result_first, result_second);
    }

  } else {
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

  return true;
}

bool LeonovaAMostDiffNeighVecElemsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace leonova_a_most_diff_neigh_vec_elems
