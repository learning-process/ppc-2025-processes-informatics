#include "gonozov_l_global_search/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <functional>
#include <limits>
#include <iostream>

#include "gonozov_l_global_search/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gonozov_l_global_search {

GonozovLGlobalSearchMPI::GonozovLGlobalSearchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool GonozovLGlobalSearchMPI::ValidationImpl() {
  return (std::get<1>(GetInput()) > 1.0) && (std::get<2>(GetInput()) < std::get<3>(GetInput())) && (std::get<4>(GetInput()) > 0);
}

bool GonozovLGlobalSearchMPI::PreProcessingImpl() {
  return true;
}

namespace {
  double Countingm(double M, double r) {
    return (M == 0.0) ? 1.0 : r * M;
  }

  double CountMIncremental(
    int t,
    double M,
    const std::vector<double>& x,
    const std::function<double(double)>& f) {

  if (M == -std::numeric_limits<double>::infinity()) {
    return std::abs((f(x[1]) - f(x[0])) / (x[1] - x[0]));
  }

  double M1 = std::abs((f(x.back()) - f(x[t - 1])) /
                       (x.back() - x[t - 1]));
  double M2 = std::abs((f(x[t]) - f(x.back())) /
                       (x[t] - x.back()));

  return std::max(M, std::max(M1, M2));
}

  double IntervalCharacteristic(double x1, double x2,
                              double f1, double f2,
                              double m) {
  double dx = x2 - x1;
  return m * dx + (f2 - f1) * (f2 - f1) / (m * dx) - 2.0 * (f1 + f2);
  }
  // template<typename Func>
  // double CountingM(int t, double M, std::vector <double> testSequence, Func function)
  // {
  //   if (M == -std::numeric_limits<double>::infinity())
  //   {
  //     M = abs((function(testSequence[1]) - function(testSequence[0])) / (testSequence[1] - testSequence[0]));
  //   }
  //   else
  //   {
  //     double M1 = abs(function(testSequence.back())- function(testSequence[t - 1])) / (testSequence.back() - testSequence[t - 1]);
  //     double M2 = abs(function(testSequence[t]) - function(testSequence.back())) / (testSequence[t] - testSequence.back());
  //     return std::max(M, std::max(M1, M2));
  //   }
  //   //M = -numeric_limits<double>::infinity();
  //   //for (int i = 1; i < testSequence.size(); i++)
  //   //{
  //   //	if (abs((testSequence[i].getZ() - testSequence[i - 1].getZ()) / (testSequence[i] - testSequence[i - 1]).getX()) > M)
  //   //	{
  //   //		M = abs((testSequence[i].getZ() - testSequence[i - 1].getZ()) / (testSequence[i] - testSequence[i - 1]).getX());
  //   //	}
  //   //}
  //   return M;
  // }

  // template<typename Func>
  // std::pair<int, double> CountingValueRt(double m, std::vector<double> local_data, Func function)
  // {
  //   std::pair<int, double> res;
  //   res.first = 1;
  //   res.second = -std::numeric_limits<double>::infinity();
  //   for (unsigned int i = 1; i < local_data.size(); i++)
  //   {
  //     double subX = (local_data[i] - local_data[i - 1]);
  //     double subZ = (function(local_data[i]) - function(local_data[i - 1]));
  //     double sumZ = (function(local_data[i]) + function(local_data[i - 1]));
  //     double interValueRt = m * subX + subZ * subZ / m / subX - 2 * sumZ;
  //     if (interValueRt > res.second)
  //     {
  //       res.second = interValueRt;
  //       res.first = i;
  //     }
  //   }
  //   return res;
  // }
}
bool GonozovLGlobalSearchMPI::RunImpl() {
  auto function = std::get<0>(GetInput());
  double r = std::get<1>(GetInput());
  double a = std::get<2>(GetInput());
  double b = std::get<3>(GetInput());
  double eps = std::get<4>(GetInput());

  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  
  // Основные переменные (хранятся только на мастере)
  
  std::vector<double> testSequence;
  double global_min_x = a;
  double global_min_value = function(a);

  int t = 1;
  double M = -std::numeric_limits<double>::infinity();
  
  if (proc_rank == 0) {
    testSequence = {a, b};
    if (function(b) < global_min_value) {
      global_min_x = b;
      global_min_value = function(b);
    }
  }

  bool continue_iteration = true;

  while (continue_iteration) {

    int n = 0;
    if (proc_rank == 0) n = testSequence.size();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

      // ---------- broadcast points ----------
    if (proc_rank != 0)
      testSequence.resize(n);

    MPI_Bcast(testSequence.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ✅ ОБЯЗАТЕЛЬНО: все процессы сортируют одинаково
    std::sort(testSequence.begin(), testSequence.end());

    // ---------- compute m ----------
    double m = 0.0;
    if (proc_rank == 0) {
      // ✅ M считается по ОТСОРТИРОВАННОМУ массиву
      M = CountMIncremental(t, M, testSequence, function);
      m = Countingm(M, r);
    }
    MPI_Bcast(&m, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int intervals = n - 1;
    int per_proc = intervals / proc_num;
    int rem = intervals % proc_num;

    int l = proc_rank * per_proc + std::min(proc_rank, rem);
    int r_i = l + per_proc + (proc_rank < rem ? 1 : 0);

    double local_max = -std::numeric_limits<double>::infinity();
    int local_idx = -1;

    if (l < r_i) {
      for (int i = l + 1; i <= r_i; ++i) {
        double R = IntervalCharacteristic(
          testSequence[i-1], testSequence[i],
          function(testSequence[i-1]),
          function(testSequence[i]),
          m
        );
        if (R > local_max) {
          local_max = R;
          local_idx = i;
        }
      }
    }

    struct { double val; int idx; } loc, glob;
    loc.val = local_max;
    loc.idx = local_idx;

    MPI_Reduce(&loc, &glob, 1, MPI_DOUBLE_INT,
               MPI_MAXLOC, 0, MPI_COMM_WORLD);

    if (proc_rank == 0) {
      if (glob.idx <= 0 || glob.idx >= static_cast<int>(testSequence.size())) {
        continue_iteration = false;
      } 
      else {
        t = glob.idx;
        if (t < 1) break;  // защита

        double x_new = 0.5 * (testSequence[t] + testSequence[t-1])
                    - (function(testSequence[t]) -
                        function(testSequence[t-1])) / (2.0 * m);

        double fx = function(x_new);
        if (fx < global_min_value) {
          global_min_value = fx;
          global_min_x = x_new;
        }
        testSequence.push_back(x_new);
      }

      continue_iteration =
        std::abs(testSequence[t] - testSequence[t-1]) > eps;
    }

    MPI_Bcast(&continue_iteration, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  }
  // do
  // {
  //   iteration_count++;
  //   // Данные для Scatterv
  //   std::vector<int> sendcounts(proc_num, 0);
  //   std::vector<int> displs(proc_num, 0);
  //   // Распределение точек, в которых были проведены испытания по процессам
  //   // int number_processed = testSequence.size() / proc_num;
  //   // int remainder = testSequence.size() % proc_num;
  //   // int local_size = number_processed + (proc_rank < remainder ? 1 : 0);

  //   double m = 0.0;
  //   if (proc_rank == 0) {
  //     M = CountingM(t, M, testSequence, function);
  //     std::sort(testSequence.begin(), testSequence.end());
  //     m = Countingm(M, r);

  //     // int offset = 0;
  //     // for (int pl = 0; pl < proc_num; pl++) {
  //     //   int p_size = number_processed + (pl < remainder ? 1 : 0);
  //     //   sendcounts[pl] = p_size;
  //     //   displs[pl] = offset;
  //     //   offset += sendcounts[pl];
  //     // }
  //   }
  //   MPI_Bcast(&m, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
  //   // Рассылаем размер последовательности
  //   int sequence_size = 0;
  //   if (proc_rank == 0) {
  //     sequence_size = static_cast<int>(testSequence.size());
  //   }
  //   MPI_Bcast(&sequence_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
  //   // Мастер вычисляет распределение данных
  //   if (proc_rank == 0) {
  //     int number_processed = sequence_size / proc_num;
  //     int remainder = sequence_size % proc_num;
      
  //     int offset = 0;
  //     for (int pl = 0; pl < proc_num; pl++) {
  //       int p_size = number_processed + (pl < remainder ? 1 : 0);
  //       sendcounts[pl] = p_size;
  //       displs[pl] = offset;
  //       offset += sendcounts[pl];
  //     }
  //   }

  //    // Рассылаем информацию о распределении
  //   MPI_Bcast(sendcounts.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);
  //   MPI_Bcast(displs.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);
    
  //   // Определяем локальный размер для текущего процесса
  //   int local_size = sendcounts[proc_rank];
    
  //   // Выделяем память для локальных данных
  //   std::vector<double> local_data;
  //   if (local_size > 0) {
  //     local_data.resize(static_cast<size_t>(local_size));
  //   }

  //   //std::vector<double> local_data(static_cast<size_t>(local_size));

  //   // MPI_Scatterv((proc_rank == 0) ? testSequence.data() : nullptr, sendcounts.data(), displs.data(),
  //   //            MPI_DOUBLE, local_data.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //   // Распределяем данные с помощью Scatterv
  //   if (proc_rank == 0) {
  //     // Мастер использует Scatterv для рассылки данных
  //     MPI_Scatterv(testSequence.data(), sendcounts.data(), displs.data(),
  //                  MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      
  //     // Мастер копирует свою часть данных в local_data
  //     if (local_size > 0) {
  //       local_data.assign(testSequence.begin() + displs[proc_rank], 
  //                        testSequence.begin() + displs[proc_rank] + local_size);
  //     }
  //   } else {
  //     // Рабочие процессы получают данные
  //     if (local_size > 0) {
  //       MPI_Scatterv(nullptr, nullptr, nullptr,
  //                    MPI_DOUBLE, local_data.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //     } else {
  //       // Если у процесса нет данных, все равно вызываем Scatterv с нулевым размером
  //       MPI_Scatterv(nullptr, nullptr, nullptr,
  //                    MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //     }
  //   }

  //   std::pair<int, double> local_max_valueRt = CountingValueRt(m, local_data, function);
  //   local_max.value = local_max_valueRt.second;
  //   local_max.rank = local_max_valueRt.first + displs[proc_rank];
    
  //   MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

  //   if (proc_rank == 0)
  //   {
  //     t = global_max.rank;
  //     double newElemSequence = 0.5 * (testSequence[t] + testSequence[t - 1]) - 0.5 / m * (function(testSequence[t]) - function(testSequence[t - 1]));
  //     if (function(newElemSequence) < global_min_value) {
  //       global_min_x = newElemSequence;
  //       global_min_value = function(newElemSequence);
  //     }
  //     testSequence.push_back(newElemSequence);

  //     // Проверяем условие остановки
  //     double interval_width = abs(testSequence[t] - testSequence[t - 1]);
  //     //continue_iteration = (interval_width > epsilon) && (iteration_count < MAX_ITERATIONS);
  //     continue_iteration = (interval_width > epsilon);
  //   }

  //   MPI_Bcast(&continue_iteration, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);    

  // } while (continue_iteration);

  // рассылаем результат всем процессам
  MPI_Bcast(&global_min_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = global_min_x;  

  return true;
}

bool GonozovLGlobalSearchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gonozov_l_global_search
