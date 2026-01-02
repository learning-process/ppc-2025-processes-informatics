#include "khruev_a_global_opt/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace khruev_a_global_opt {

struct IntervalInfo {
  double R;
  int index;  // индекс в массиве trials_ (указывает на правый конец интервала)

  bool operator>(const IntervalInfo &other) const {
    return R > other.R;  // Для сортировки по убыванию
  }
};

KhruevAGlobalOptMPI::KhruevAGlobalOptMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KhruevAGlobalOptMPI::ValidationImpl() {
  return GetInput().max_iter > 0 && GetInput().epsilon > 0 && GetInput().r > 1.0;
}

bool KhruevAGlobalOptMPI::PreProcessingImpl() {
  trials_.clear();
  return true;
}

double KhruevAGlobalOptMPI::CalculateFunction(double t) {
  double u, v;
  d2xy(t, u, v);
  double real_x = GetInput().ax + u * (GetInput().bx - GetInput().ax);
  double real_y = GetInput().ay + v * (GetInput().by - GetInput().ay);
  return target_function(GetInput().func_id, real_x, real_y);
}

void KhruevAGlobalOptMPI::AddTrialUnsorted(double t, double z) {
  Trial tr;
  tr.x = t;
  tr.z = z;
  trials_.push_back(tr);
}

bool KhruevAGlobalOptMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Начальные испытания
  if (trials_.empty()) {
    AddTrialUnsorted(0.0, CalculateFunction(0.0));
    AddTrialUnsorted(1.0, CalculateFunction(1.0));
  }
  std::sort(trials_.begin(), trials_.end());

  int k = 2;  // Уже провели 2 испытания
  // bool stop_flag = false;

  while (k < GetInput().max_iter) {
    // 1. Вычисляем M (одинаково на всех процессах)
    double max_slope = 0.0;
    for (size_t i = 1; i < trials_.size(); ++i) {
      double dx = trials_[i].x - trials_[i - 1].x;
      double dz = std::abs(trials_[i].z - trials_[i - 1].z);
      if (dx > 1e-12) {
        max_slope = std::max(max_slope, dz / dx);
      }
    }

    double M = (max_slope > 0) ? GetInput().r * max_slope : 1.0;

    // 2. Вычисляем характеристики R
    std::vector<IntervalInfo> intervals;
    intervals.reserve(trials_.size() - 1);
    for (size_t i = 1; i < trials_.size(); ++i) {
      double dx = trials_[i].x - trials_[i - 1].x;
      double z_r = trials_[i].z;
      double z_l = trials_[i - 1].z;

      // Классическая формула Стронгина
      double R = M * dx + ((z_r - z_l) * (z_r - z_l)) / (M * dx) - 2.0 * (z_r + z_l);
      intervals.push_back({R, (int)i});
    }

    // 3. Сортируем интервалы по убыванию характеристики
    std::sort(intervals.begin(), intervals.end(), std::greater<IntervalInfo>());

    // 4. Проверка критерия остановки (по самому широкому из лучших интервалов)
    bool local_stop = true;
    int num_to_check = std::min((int)intervals.size(), size);
    for (int i = 0; i < num_to_check; ++i) {
      int idx = intervals[i].index;
      if (trials_[idx].x - trials_[idx - 1].x > GetInput().epsilon) {
        local_stop = false;
        break;
      }
    }

    // Синхронная остановка
    int stop_signal = 0;
    int local_stop_signal = local_stop ? 1 : 0;
    MPI_Allreduce(&local_stop_signal, &stop_signal, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (stop_signal) {
      break;
    }

    // 5. Генерация новых точек
    struct Point {
      double x, z;
    };
    Point my_point = {-1.0, 0.0};

    if (rank < (int)intervals.size()) {
      int idx = intervals[rank].index;
      double x_r = trials_[idx].x;
      double x_l = trials_[idx - 1].x;
      double z_r = trials_[idx].z;
      double z_l = trials_[idx - 1].z;

      // Формула вычисления новой точки испытания
      double new_x = 0.5 * (x_r + x_l) - (z_r - z_l) / (2.0 * M);

      // Защита от выхода за границы интервала
      if (new_x <= x_l || new_x >= x_r) {
        new_x = 0.5 * (x_r + x_l);
      }

      my_point.x = new_x;
      my_point.z = CalculateFunction(new_x);
    }

    // 6. Сбор результатов со всех процессов
    std::vector<Point> global_res(size);
    MPI_Allgather(&my_point, 2, MPI_DOUBLE, global_res.data(), 2, MPI_DOUBLE, MPI_COMM_WORLD);

    // 7. Обновление списка
    for (int i = 0; i < size; ++i) {
      if (global_res[i].x >= 0.0) {
        AddTrialUnsorted(global_res[i].x, global_res[i].z);
        k++;
      }
    }
    std::sort(trials_.begin(), trials_.end());
  }

  // Поиск финального минимума среди всех испытаний
  auto it = std::min_element(trials_.begin(), trials_.end(), [](const Trial &a, const Trial &b) { return a.z < b.z; });

  double u, v;
  d2xy(it->x, u, v);
  result_.x = GetInput().ax + u * (GetInput().bx - GetInput().ax);
  result_.y = GetInput().ay + v * (GetInput().by - GetInput().ay);
  result_.value = it->z;
  result_.iter_count = k;

  GetOutput() = result_;
  return true;
}

bool KhruevAGlobalOptMPI::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_global_opt
