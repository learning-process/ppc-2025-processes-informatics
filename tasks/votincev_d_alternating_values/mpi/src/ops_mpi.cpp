#include "votincev_d_alternating_values/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "util/include/util.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"

namespace votincev_d_alternating_values {

// это у всех одинаковое
VotincevDAlternatingValuesMPI::VotincevDAlternatingValuesMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

// проверка данных на адекватность
bool VotincevDAlternatingValuesMPI::ValidationImpl() {
  return !(GetInput().empty());
}

// препроцессинг (например в КГ)
bool VotincevDAlternatingValuesMPI::PreProcessingImpl() {
  v = GetInput();
  return true;
}

bool VotincevDAlternatingValuesMPI::RunImpl() {
  double start_time = 0, end_time = 0;

  start_time = MPI_Wtime();
  int allSwaps = 0;
  // получаю кол-во процессов
  int ProcessN;
  MPI_Comm_size(MPI_COMM_WORLD, &ProcessN);

  // получаю ранг процесса
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  size_t partSize;
  if (ProcRank == 0) {
    size_t base = v.size() / ProcessN;    // минимум на обработку
    size_t remain = v.size() % ProcessN;  // остаток (распределим)

    size_t startId = 0;
    for (int i = 1; i < ProcessN; i++) {
      partSize = base;
      if (remain) {  // если есть остаток - то распределяем между первыми
        partSize++;
        remain--;
      }

      partSize++;  // цепляем правого соседа, 0-й будет последним - поэтому он будет последний кусок считать

      // Вместо пересылки данных - пересылаем индексы начала и конца
      int64_t indices[2] = {static_cast<int64_t>(startId), static_cast<int64_t>(startId + partSize)};
      MPI_Send(indices, 2, MPI_INT64_T, i, 0, MPI_COMM_WORLD);
      // std::cout << "Id: " << startId << '\n';

      startId += partSize - 1;

      // вычисляю для последнего
      if (i == (ProcessN - 1)) {
        partSize = base + remain;
      }
    }

    for (size_t j = startId + 1; j < v.size(); j++) {
      if ((v[j - 1] < 0 && v[j] >= 0) || (v[j - 1] >= 0 && v[j] < 0)) {
        allSwaps++;
      }
    }

    // получаю посчитанные swap от процессов
    for (int i = 1; i < ProcessN; i++) {
      int tmp;
      // что , сколько, тип, кому, тег, коммуникатор
      MPI_Recv(&tmp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      allSwaps += tmp;
    }

    // на этом этапе 0-й процесс сделал всю работу
  }

  for (int i = 1; i < ProcessN; i++) {
    if (ProcRank == i) {
      // получаем индексы вместо данных
      int64_t indices[2];
      MPI_Recv(indices, 2, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      size_t start_index = static_cast<size_t>(indices[0]);
      size_t end_index = static_cast<size_t>(indices[1]);

      // корректируем конечный индекс если нужно
      if (end_index > v.size()) {
        end_index = v.size();
      }

      int swapCount = 0;
      // обрабатываем свой диапазон из глобального вектора v
      for (size_t j = start_index + 1; j < end_index; j++) {
        if ((v[j - 1] < 0 && v[j] >= 0) || (v[j - 1] >= 0 && v[j] < 0)) {
          swapCount++;
        }
      }

      MPI_Send(&swapCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end_time = MPI_Wtime();

  if (ProcRank == 0) {
    // отправляем всем процессам корректный результат
    for (int i = 1; i < ProcessN; i++) {
      MPI_Send(&allSwaps, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    // сами устанавливаем значение
    GetOutput() = allSwaps;
  }
  for (int i = 1; i < ProcessN; i++) {
    if (ProcRank == i) {
      int allSw;
      MPI_Recv(&allSw, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      GetOutput() = allSw;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (ProcRank == 0) {
    std::cout << "MPI_was_working_" << (end_time - start_time) << "\n";
  }

  return true;
}

// удобно возвращаем данные (???)
bool VotincevDAlternatingValuesMPI::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_alternating_values
