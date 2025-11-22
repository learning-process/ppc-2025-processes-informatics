#include "shvetsova_k_max_diff_neig_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_max_diff_neig_vec {

ShvetsovaKMaxDiffNeigVecMPI::ShvetsovaKMaxDiffNeigVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::pair<double, double>{0.0, 0.0};
}

bool ShvetsovaKMaxDiffNeigVecMPI::ValidationImpl() {
  return true;
}

bool ShvetsovaKMaxDiffNeigVecMPI::PreProcessingImpl() {
  data = GetInput();
  return true;
}

bool ShvetsovaKMaxDiffNeigVecMPI::RunImpl() {
  int CountOfProc, rank;
  int SizeOfVector = data.size();
  MPI_Comm_size(MPI_COMM_WORLD, &CountOfProc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(&SizeOfVector, 1, MPI_INT, 0, MPI_COMM_WORLD);  // сказали всем процессам, какой размер у вектора

  if (SizeOfVector < 2) {
    GetOutput() = {0.0, 0.0};
    return true;
  }

  std::vector<int> countElems(CountOfProc, 0);  // количсетво элементов на каждый из процессов
  std::vector<int> ind(CountOfProc, 0);         // индекс, начиная с которого элементы принадлежат процессам

  if (rank == 0) {
    int base = SizeOfVector / CountOfProc;
    int rem = SizeOfVector % CountOfProc;
    int sum = 0;

    for (int i = 0; i < CountOfProc; i++) {
      countElems[i] = base + (i < rem ? 1 : 0);
      ind[i] = sum;
      sum += countElems[i];
    }
  }

  MPI_Bcast(countElems.data(), CountOfProc, MPI_INT, 0,
            MPI_COMM_WORLD);  // разослали процессам информацию о кол-ве элементов
  MPI_Bcast(ind.data(), CountOfProc, MPI_INT, 0,
            MPI_COMM_WORLD);  // разослали порцессам индексы, с которых начинаются их элементы

  int partSize = countElems[rank];     // определяяем кол-во элементов для текущего процесса
  std::vector<double> part(partSize);  // создаем вектор с размером = кол-во элементов для процесса

  MPI_Scatterv(data.data(), countElems.data(), ind.data(), MPI_DOUBLE, part.data(), partSize, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);  // распилили вектор

  // ищем локальные результаты
  double localDiff = -1.0;
  double localA = 0.0, localB = 0.0;

  for (int i = 0; i < partSize - 1; i++) {
    double diff = std::abs(part[i] - part[i + 1]);
    if (diff > localDiff) {
      localDiff = diff;
      localA = part[i];
      localB = part[i + 1];
    }
  }

  if (localDiff < 0) {
    localDiff = 0.0;
  }

  // обрабатиываем то, что на границах кусочков в процессах
  if (CountOfProc > 1) {
    if (rank < CountOfProc - 1 && partSize > 0) {
      double lastCurr = part[partSize - 1];
      MPI_Send(&lastCurr, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (rank > 0) {
      double lastPrev = 0;
      MPI_Recv(&lastPrev, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (partSize > 0) {
        double diff = std::abs(lastPrev - part[0]);
        if (diff > localDiff) {
          localDiff = diff;
          localA = lastPrev;
          localB = part[0];
        }
      }
    }
  }

  // собираем локальные рузультаты
  std::vector<double> allDiffs(CountOfProc);
  MPI_Gather(&localDiff, 1, MPI_DOUBLE, allDiffs.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // на 0 процессе вычисляем глобальный результат
  double globalDiff = 0.0;
  int winner_rank = 0;

  if (rank == 0) {
    globalDiff = *std::max_element(allDiffs.begin(), allDiffs.end());

    for (int i = 0; i < CountOfProc; i++) {
      if (std::abs(allDiffs[i] - globalDiff) < 1e-12) {
        winner_rank = i;
        break;
      }
    }
  }

  MPI_Bcast(&winner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);  // рассылаем номер процесса победителя

  std::vector<double> allPairs(CountOfProc * 2);
  double localPair[2] = {localA, localB};

  MPI_Gather(localPair, 2, MPI_DOUBLE, allPairs.data(), 2, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);  // получаем все локальные пары результаты (где максимальная разница на прооцессе)

  double resultPair[2] = {0.0, 0.0};

  if (rank == 0) {
    // вычисляем глобальную пару с максимальной разницей
    resultPair[0] = allPairs[winner_rank * 2];
    resultPair[1] = allPairs[winner_rank * 2 + 1];
  }

  // рассылаем результат всем процессам
  MPI_Bcast(resultPair, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = {resultPair[0], resultPair[1]};

  return true;
}

bool ShvetsovaKMaxDiffNeigVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_max_diff_neig_vec
