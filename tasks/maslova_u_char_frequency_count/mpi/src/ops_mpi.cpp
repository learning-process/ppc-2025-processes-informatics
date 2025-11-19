#include "maslova_u_char_frequency_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>
#include <iostream>

#include "maslova_u_char_frequency_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace maslova_u_char_frequency_count {

MaslovaUCharFrequencyCountMPI::MaslovaUCharFrequencyCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MaslovaUCharFrequencyCountMPI::ValidationImpl() {
  return true;
}

bool MaslovaUCharFrequencyCountMPI::PreProcessingImpl() {
  return true;
}

bool MaslovaUCharFrequencyCountMPI::RunImpl() {
  int rank = 0;
  int proc_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // id процесса
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size); // количество процессов

  std::string input_string;
  char input_char = 0;
  size_t input_str_size = 0;

  if (rank == 0) { 
    input_string = GetInput().first;
    input_char = GetInput().second;
    input_str_size = input_string.size(); // получили данные
    
    if (input_string.empty()) {
      GetOutput() = 0; //если строка пустая, выводим сразу 0
    }
  }

  MPI_Bcast(&input_str_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD); // отправляем размер строки
  if (input_str_size == 0) {
    return true; 
  }

  MPI_Bcast(&input_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD); // отправляем нужный символ
  
  std::vector<int> send_counts(proc_size); //здесь размеры всех порций
  std::vector<int> displs(proc_size); //смещения
  if (rank == 0) {
    int part = input_str_size / proc_size;
    int rem = input_str_size % proc_size;
    for (int i = 0; i < proc_size; ++i) {
      send_counts[i] = part + (i < rem ? 1 : 0); //общий размер, включающий остаток, если он входит
    }
    displs[0] = 0;
    for (size_t i = 1; i < proc_size; ++i) {
      displs[i] = displs[i-1] + send_counts[i-1];
    }
  }

  MPI_Bcast(send_counts.data(), proc_size, MPI_INT, 0, MPI_COMM_WORLD); //отправляем размеры порций
  std::vector<char> local_str(send_counts[rank]);
  MPI_Scatterv(
      (rank == 0) ? input_string.data() : nullptr,
      send_counts.data(), displs.data(), MPI_CHAR,
      local_str.data(), local_str.size(), MPI_CHAR,
      0, MPI_COMM_WORLD //распределяем данные
  );

  size_t local_count = std::count(local_str.begin(), local_str.end(), input_char);

  size_t global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD); //собрали данные со всех процессов
  
  GetOutput() = global_count; //вывели результат

  return true;
}

bool MaslovaUCharFrequencyCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_char_frequency_count
