#include "ovsyannikov_n_num_mistm_in_two_str/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace ovsyannikov_n_num_mistm_in_two_str {

OvsyannikovNNumMistmInTwoStrMPI::OvsyannikovNNumMistmInTwoStrMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OvsyannikovNNumMistmInTwoStrMPI::ValidationImpl() {
  int proc_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  if (proc_rank == 0) {
    return GetInput().first.size() == GetInput().second.size();
  }
  return true;
}

bool OvsyannikovNNumMistmInTwoStrMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool OvsyannikovNNumMistmInTwoStrMPI::RunImpl() {
  int proc_rank;
  int proc_num;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  int total_len = 0;
  if (proc_rank == 0) {
    total_len = static_cast<int>(GetInput().first.size());
  }

  MPI_Bcast(&total_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_len == 0) {
    return true;
  }

  std::vector<int> elems_per_proc(proc_num);
  std::vector<int> shifts(proc_num);

  int tail = total_len % proc_num;
  int accum = 0;
  for (int i = 0; i < proc_num; i++) {
    elems_per_proc[i] = (total_len / proc_num) + (i < tail ? 1 : 0);
    shifts[i] = accum;
    accum += elems_per_proc[i];
  }

  // Подготовка данных
  std::vector<char> main_buff;
  std::vector<int> byte_counts(proc_num);
  std::vector<int> byte_shifts(proc_num);

  if (proc_rank == 0) {
    main_buff.resize(2 * total_len);
    const auto &seq_one = GetInput().first;
    const auto &seq_two = GetInput().second;

    int iter_pos = 0;

    // Собираем данные
    for (int i = 0; i < proc_num; ++i) {
      int part_len = elems_per_proc[i];
      int read_from = shifts[i];

      // Копируем кусок 1 строки
      if (part_len > 0) {
        std::copy(seq_one.begin() + read_from, seq_one.begin() + read_from + part_len, main_buff.begin() + iter_pos);
      }
      // Копируем кусок 2 строки
      if (part_len > 0) {
        std::copy(seq_two.begin() + read_from, seq_two.begin() + read_from + part_len,
                  main_buff.begin() + iter_pos + part_len);
      }

      byte_counts[i] = 2 * part_len;
      byte_shifts[i] = iter_pos;

      iter_pos += (2 * part_len);
    }
  }

  // Прием данных
  int my_chunk = elems_per_proc[proc_rank];
  std::vector<char> local_store(2 * my_chunk);

  MPI_Scatterv(main_buff.data(), byte_counts.data(), byte_shifts.data(), MPI_CHAR, local_store.data(), 2 * my_chunk,
               MPI_CHAR, 0, MPI_COMM_WORLD);

  int priv_err_cnt = 0;
  for (int i = 0; i < my_chunk; ++i) {
    // Сравниваем элемент из первой половины с элементом из второй половины
    if (local_store[i] != local_store[my_chunk + i]) {
      priv_err_cnt++;
    }
  }

  // Сборка результата
  int total_err_cnt = 0;
  MPI_Allreduce(&priv_err_cnt, &total_err_cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = total_err_cnt;
  return true;
}

bool OvsyannikovNNumMistmInTwoStrMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_num_mistm_in_two_str
