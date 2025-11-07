#include "votincev_d_alternating_values/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <vector>

#include "votincev_d_alternating_values/common/include/common.hpp"

namespace votincev_d_alternating_values {

VotincevDAlternatingValuesMPI::VotincevDAlternatingValuesMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

// проверка данных на адекватность
bool VotincevDAlternatingValuesMPI::ValidationImpl() {
  return true;
}

// препроцессинг (например в КГ)
bool VotincevDAlternatingValuesMPI::PreProcessingImpl() {
  vect_data_ = GetInput();
  GetOutput() = -1;  // специальное значение
  return true;
}

// код MPI
bool VotincevDAlternatingValuesMPI::RunImpl() {
  // double start_time = 0, end_time = 0;
  // start_time = MPI_Wtime();

  int all_swaps = 0;

  int process_n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &process_n);  // получаю кол-во процессов

  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);  // получаю ранк процесса

  // если процессов больше, чем размер вектора
  const int vector_size = static_cast<int>(vect_data_.size());
  process_n = std::min(vector_size, process_n);

  if (proc_rank == 0 && process_n > 0) {
    all_swaps = ProcessMaster(process_n);  // главный процесс (распределяет + считает часть)
  } else if (proc_rank < process_n) {
    ProcessWorker();  // процессы-рабочие (только считают)
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // end_time = MPI_Wtime();
  // if(proc_rank == 0) {
  //   std::cout << "MPI was working: " << end_time-start_time << "\n";
  // }

  SyncResults(all_swaps);  // посылаю результат всем процессам
  return true;
}

// работа 0-го процесса
int VotincevDAlternatingValuesMPI::ProcessMaster(int process_n) {
  int all_swaps = 0;
  const int vector_size = static_cast<int>(vect_data_.size());
  const int base = vector_size / process_n;
  int remain = vector_size % process_n;
  int start_id = 0;

  // распределяем работу между процессами
  for (int i = 1; i < process_n; i++) {
    int part_size = base;
    if (remain != 0) {
      part_size++;
      remain--;
    }
    part_size++;

    // пересылаю индекс начала и конца процессу ранка i
    SendRangeToWorker(start_id, start_id + part_size, i);
    start_id += part_size - 1;
  }

  // считаем свою часть
  all_swaps += CountSwaps(start_id + 1, vector_size);

  // собираем результаты с процессов-работников
  all_swaps += CollectWorkerResults(process_n);
  return all_swaps;
}

// работа 1,2.. N-1 процессов
void VotincevDAlternatingValuesMPI::ProcessWorker() {
  std::array<int, 2> indices = ReceiveRange();
  int swap_count = CountSwaps(indices[0] + 1, indices[1]);
  SendResult(swap_count);
}

// отправка части вектора на обработку рабочему процессу
void VotincevDAlternatingValuesMPI::SendRangeToWorker(int start, int end, int worker) {
  std::array<int, 2> indices{start, end};
  MPI_Send(indices.data(), 2, MPI_INT, worker, 0, MPI_COMM_WORLD);
}

// рабочие процессы получают данные
std::array<int, 2> VotincevDAlternatingValuesMPI::ReceiveRange() {
  std::array<int, 2> indices{0, 0};
  MPI_Recv(indices.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  return {indices[0], indices[1]};
}

// подсчет числа чередований знаков между соседними элементами вектора (его части)
int VotincevDAlternatingValuesMPI::CountSwaps(int start, int end) {
  int count = 0;
  const int vector_size = static_cast<int>(vect_data_.size());
  const int safe_end = std::min(end, vector_size);

  for (int j = start; j < safe_end; j++) {
    if ((vect_data_[j - 1] < 0 && vect_data_[j] >= 0) || (vect_data_[j - 1] >= 0 && vect_data_[j] < 0)) {
      count++;
    }
  }
  return count;
}

// сбор с процессов-работников сколько каждый насчитал чередований
int VotincevDAlternatingValuesMPI::CollectWorkerResults(int process_n) {
  int total = 0;
  for (int i = 1; i < process_n; i++) {
    int tmp = 0;
    MPI_Recv(&tmp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    total += tmp;
  }
  return total;
}

// для отправки результата (кол-во чередований, которое насчитал рабочий процесс)
void VotincevDAlternatingValuesMPI::SendResult(int result) {
  MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

// 0й процесс делится результатом с процессами-работниками
void VotincevDAlternatingValuesMPI::SyncResults(int &all_swaps) {
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&all_swaps, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = all_swaps;
  MPI_Barrier(MPI_COMM_WORLD);
}

bool VotincevDAlternatingValuesMPI::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_alternating_values
