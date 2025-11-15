#include "gasenin_l_int_rec_meth/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <string>
#include <vector>

#include "gasenin_l_int_rec_meth/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_int_rec_meth {

GaseninLIntRecMethMPI::GaseninLIntRecMethMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GaseninLIntRecMethMPI::ValidationImpl() {
  const auto &[str1, str2] = GetInput();
  return str1.length() <= 10000 && str2.length() <= 10000;
}

bool GaseninLIntRecMethMPI::PreProcessingImpl() {
  return true;
}

bool GaseninLIntRecMethMPI::RunImpl() {
  const auto &[str1, str2] = GetInput();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Для лексикографического сравнения нам нужно найти ПЕРВОЕ различие
  // Поэтому используем другой подход

  size_t total_len = std::max(str1.length(), str2.length());

  if (total_len == 0) {
    // Обе строки пустые
    GetOutput() = 0;
    return true;
  }

  // Каждый процесс проверяет свою часть строки на наличие первого различия
  size_t chunk_size = (total_len + size - 1) / size;
  size_t start = rank * chunk_size;
  size_t end = std::min(start + chunk_size, total_len);

  int local_result = 0;
  size_t local_diff_pos = total_len;  // позиция различия для этого процесса

  // Ищем первое различие в своей части
  for (size_t i = start; i < end && local_result == 0; ++i) {
    char c1 = (i < str1.length()) ? str1[i] : '\0';
    char c2 = (i < str2.length()) ? str2[i] : '\0';

    if (c1 != c2) {
      local_diff_pos = i;
      local_result = (c1 < c2) ? -1 : 1;
      break;
    }
  }

  // Собираем информацию о самом раннем различии со всех процессов
  struct DiffInfo {
    size_t pos;
    int result;
  };

  DiffInfo local_info = {local_diff_pos, local_result};
  std::vector<DiffInfo> all_infos;

  if (rank == 0) {
    all_infos.resize(size);
  }

  MPI_Gather(&local_info, sizeof(DiffInfo), MPI_BYTE, all_infos.data(), sizeof(DiffInfo), MPI_BYTE, 0, MPI_COMM_WORLD);

  int final_result = 0;

  if (rank == 0) {
    // Находим самое раннее различие среди всех процессов
    size_t earliest_pos = total_len;

    for (const auto &info : all_infos) {
      if (info.result != 0 && info.pos < earliest_pos) {
        earliest_pos = info.pos;
        final_result = info.result;
      }
    }

    // Если различий не найдено, сравниваем длины
    if (final_result == 0) {
      if (str1.length() < str2.length()) {
        final_result = -1;
      } else if (str1.length() > str2.length()) {
        final_result = 1;
      } else {
        final_result = 0;
      }
    }

    GetOutput() = final_result;
  }

  // Рассылаем результат всем процессам
  MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool GaseninLIntRecMethMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_int_rec_meth
