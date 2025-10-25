#include "votincev_d_alternating_values/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "util/include/util.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"

namespace votincev_d_alternating_values {

// это у всех одинаковое
VotincevDAlternatingValuesMPI::VotincevDAlternatingValuesMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // установка типа задачи
  GetInput() = in;                       // получаем данные
  // GetOutput() = 0; // у меня vector<double> как будто =0 не сработает
}

// проверяет входные данные на адекватность
bool VotincevDAlternatingValuesMPI::ValidationImpl() {
  return !GetInput().empty();
}

// нужно например в КГ (препроцессинг изображения)
bool VotincevDAlternatingValuesMPI::PreProcessingImpl() {
  return true;
}

// здесь реализуем MPI
bool VotincevDAlternatingValuesMPI::RunImpl() {
  std::vector<double> v = GetInput();
  int allSwaps = 0;  // GetOutput()
                     // получаю кол-во процессов
  int ProcessN;
  MPI_Comm_size(MPI_COMM_WORLD, &ProcessN);

  // получаю ранг процесса
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  int partSize;
  if (ProcRank == 0) {
    int base = v.size() / ProcessN;    // минимум на обработку
    int remain = v.size() % ProcessN;  // остаток (распределим)

    int startId = 0;
    for (int i = 1; i < ProcessN; i++) {
      partSize = base;
      if (remain) {  // если есть остаток - то распределяем между первыми
        partSize++;
        remain--;
      }

      partSize++;  // цепляем правого соседа, 0-й будет последним - поэтому он будет последний кусок считать

      // отправляем всем процессам их размер
      MPI_Send(&partSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      // std::cout << "Id: " << startId << '\n';

      // отправлем всем прцоессам их кусок вектора
      MPI_Send(v.data() + startId, partSize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      startId += partSize - 1;

      // вычисляю для последнего
      if (i == ProcessN - 1) {
        partSize = base + remain;
      }
    }

    for (int j = startId + 1; j < v.size(); j++) {
      if (v[j - 1] < 0 && v[j] >= 0 || v[j - 1] >= 0 && v[j] < 0) {
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
      std::vector<double> vectData;
      // отправляю partSize процессам кроме последнего
      MPI_Recv(&partSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      vectData.resize(partSize);
      MPI_Recv(vectData.data(), partSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int swapCount = 0;
      for (int j = 1; j < vectData.size(); j++) {
        if (vectData[j - 1] < 0 && vectData[j] >= 0 || vectData[j - 1] >= 0 && vectData[j] < 0) {
          swapCount++;
        }
      }
      MPI_Send(&swapCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }
  GetOutput() = allSwaps;
  return true;
}

// данные возвращаем в удобном формате (для себя)
bool VotincevDAlternatingValuesMPI::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_alternating_values
