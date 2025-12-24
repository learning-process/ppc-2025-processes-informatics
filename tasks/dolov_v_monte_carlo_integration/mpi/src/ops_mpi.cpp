#include "dolov_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <random>
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"
#include "util/include/util.hpp"

namespace dolov_v_monte_carlo_integration {

DolovVMonteCarloIntegrationMPI::DolovVMonteCarloIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool DolovVMonteCarloIntegrationMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid = true;

  // Валидация происходит только на корневом процессе (rank 0)
  if (rank == 0) {
    const auto &in = GetInput();
    is_valid = in.func && (in.samples_count > 0) && (in.dimension > 0) &&
               (in.center.size() == static_cast<size_t>(in.dimension)) && (in.radius > 0.0);
  }

  // Результат валидации должен быть передан всем процессам
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool DolovVMonteCarloIntegrationMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DolovVMonteCarloIntegrationMPI::RunImpl() {
  int current_rank = 0;
  int total_procs = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total_procs);

  InType params = GetInput();

  int dim = params.dimension;
  int total_samples = params.samples_count;
  double rad = params.radius;
  int domain_type_int = static_cast<int>(params.domain_type);

  // Bcast скалярных параметров
  MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rad, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&domain_type_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Принимаем bcastнутые значения на всех процессах
  if (current_rank != 0) {
    params.dimension = dim;
    params.samples_count = total_samples;
    params.radius = rad;
    params.domain_type = static_cast<IntegrationDomain>(domain_type_int);

    params.center.resize(dim);
  }

  // Bcast вектора центра
  if (dim > 0) {
    MPI_Bcast(params.center.data(), dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  int local_samples = total_samples / total_procs;
  // Добавляем остаток к последнему процессу
  if (current_rank == total_procs - 1) {
    local_samples += total_samples % total_procs;
  }

  const double R_sq = params.radius * params.radius;

  std::mt19937 random_generator(current_rank + 101);
  std::uniform_real_distribution<double> value_distributor(-params.radius, params.radius);

  double local_sum_of_f = 0.0;
  std::vector<double> sample_point(params.dimension);

  for (int i = 0; i < local_samples; ++i) {
    double distance_sq = 0.0;
    bool is_valid_point = true;

    // Генерация случайной точки в гиперкубе
    for (int d = 0; d < params.dimension; ++d) {
      sample_point[d] = params.center[d] + value_distributor(random_generator);
    }

    if (params.domain_type == IntegrationDomain::kHyperSphere) {
      // Проверка попадания в гиперсферу
      for (int d = 0; d < params.dimension; ++d) {
        distance_sq += std::pow(sample_point[d] - params.center[d], 2);
      }

      if (distance_sq > R_sq) {
        is_valid_point = false;
      }
    }

    if (is_valid_point) {
      local_sum_of_f += params.func(sample_point);
    }
  }

  double global_sum_of_f = 0.0;
  // Собираем все локальные суммы на корневой процесс 0
  MPI_Reduce(&local_sum_of_f, &global_sum_of_f, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double final_integral_result = 0.0;

  if (current_rank == 0) {
    // Объем описанного гиперкуба
    const double hyperspace_volume = std::pow(2.0 * params.radius, params.dimension);
    final_integral_result = hyperspace_volume * (global_sum_of_f / total_samples);
  }

  // Bcast финальный результат всем процессам, чтобы GetOutput() был одинаковым
  MPI_Bcast(&final_integral_result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = final_integral_result;

  // Проверка на корректность результата
  return std::isfinite(GetOutput());
}

bool DolovVMonteCarloIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dolov_v_monte_carlo_integration
