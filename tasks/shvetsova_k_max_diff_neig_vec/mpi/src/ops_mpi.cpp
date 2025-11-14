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
  GetOutput() = std::pair<double, std::pair<int, int>>{{0}, {0, 0}};
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

  MPI_Bcast(&SizeOfVector, 1, MPI_INT, 0, MPI_COMM_WORLD);  // всем процессам отсылается размер вектора

  if (SizeOfVector < 2) {
    return true;
  }

  std::vector<int> countElems(CountOfProc, 0);
  std::vector<int> ind(CountOfProc, 0);

  // разделение работы между процессами
  if (rank == 0) {
    int base = SizeOfVector / CountOfProc;
    int rem = SizeOfVector % CountOfProc;
    int sum = 0;
    for (int i = 0; i < CountOfProc; i++) {
      countElems[i] += base;
      if (i < rem) {
        countElems[i] += 1;
      }
      ind[i] = sum;
      sum += countElems[i];
    }
  }

  MPI_Bcast(countElems.data(), CountOfProc, MPI_INT, 0,
            MPI_COMM_WORLD);  // процессам говорим, сколько элементов им принадлежит
  MPI_Bcast(ind.data(), CountOfProc, MPI_INT, 0,
            MPI_COMM_WORLD);  // процессам говорим, с какого инджекса считать свое количсетво элементов

  int PeaceSize = countElems[rank];      // инициализируем размер кусочка вектора
  std::vector<double> peace(PeaceSize);  // создаем векторр - кусочек от исходного

  MPI_Scatterv(GetInput().data(), countElems.data(), ind.data(), MPI_DOUBLE, peace.data(), PeaceSize, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);  // распилили вектор

  // работа процесса
  double LocalMx = 0;
  int LocInd = 0;
  for (int i = 0; i < PeaceSize - 1; i++) {
    double tmp = std::abs(peace[i] - peace[i + 1]);
    if (LocalMx <= tmp) {
      LocalMx = tmp;
      LocInd = ind[rank] + i;  // вычисляется глобальный индекс элемента - начало кусочка + текущий индекс элемента = i
    }
  }
  // Обработка граничных эелеметов
  if (rank > 0) {
    double PrevLast = 0;
    MPI_Recv(&PrevLast, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    double BoundDiff = std::abs(PrevLast - peace[0]);
    int GlobalBoundInd = ind[rank] - 1;
    if (LocalMx <= BoundDiff) {
      LocalMx = BoundDiff;
      LocInd = GlobalBoundInd;
    }
  }

  // пересылаем последний элемент предыдущего процесса следующему
  if (rank < CountOfProc - 1) {
    double last = peace.back();
    MPI_Send(&last, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
  }

  std::pair<double, int> ValInd{LocalMx, LocInd};

  MPI_Reduce(&ValInd.first, &ValInd.second, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0,
             MPI_COMM_WORLD);  // MPIMAXLOC выбирает максимум из всех результатов процессов

  // Запись результата 0 процессом
  if (rank == 0) {
    GetOutput().first = ValInd.first;
    GetOutput().second.first = ValInd.second;
    GetOutput().second.second = ValInd.second + 1;
  }

  return true;
}

bool ShvetsovaKMaxDiffNeigVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_max_diff_neig_vec
