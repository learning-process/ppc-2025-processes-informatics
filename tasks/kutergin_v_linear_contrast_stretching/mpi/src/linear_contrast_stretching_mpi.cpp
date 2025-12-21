#include "../include/linear_contrast_stretching_mpi.hpp"

#include <mpi.h>
#include <algorithm>
#include <vector>
#include <cmath>

#include "../../common/include/common.hpp"

namespace kutergin_v_linear_contrast_stretching
{

LinearContrastStretchingMPI::LinearContrastStretchingMPI(const InType &in) 
{
  SetTypeOfTask(GetStaticTypeOfTask());  // установка типа задачи
  GetInput() = in;                       // сохранение входных данных
  GetOutput().resize(in.data.size());    // инициализация выходных данных                  
}

bool LinearContrastStretchingMPI::ValidationImpl() 
{
    const auto& img_data = GetInput().data;
    if (img_data.empty()) // пустое изображение невалидно
    {
        return false;
    }

    int process_count = 1;
    if (ppc::util::IsUnderMpirun())
    {
        MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    }

    return (img_data.size() % process_count == 0); // общее число пикселей делится нацело на число процессов
}

bool LinearContrastStretchingMPII::PreProcessingImpl() 
{
  return true;
}

bool LinearContrastStretchingMPI::RunImpl() 
{
    int process_rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    const auto& image_in_full = GetInput().data;
    auto& image_out_full = GetOutput();

    const int chunk_size = image_in_full.size() / process_count; // число пикселей на процесс
    std::vector<unsigned char> local_chunk(chunk_size); // локальный буфер для приема части вектора на каждом процессе

    MPI_Scatter(image_in_full.data(), chunk_size, MPI_UNSIGNED_CHAR, local_chunk.data(), chunk_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // поиск локальных минимумов и максимумов (на каждом куске вектора)
    const auto& minmax_it = std::minmax_element(local_chunk.begin(), local_chunk.end());
    unsigned char local_min = *minmax_it.first;
    unsigned char local_max = *minmax_it.second;

    // получение глобального минимума и максимума (для всего вектора)
    unsigned char global_min = 0;
    unsigned char global_max = 0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD); // сбор минимума из всех local_min
    MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD); // сбор максимума из всех local_max

    if (global_min == global_max)
    {
      // ничего не делаем, local_chunk содержит уже правильные пиксели
    }
    else 
    {
      const double scale = 255.0 / (global_max - global_min);
      for (size_t i = 0; i < local_chunk.size(); i++)
      {
        double p_out = (local_chunk[i] - global_min) * scale;
        local_chunk[i] = static_cast<unsigned char>(std::round(p_out));
      }
    }

    MPI_Gather(local_chunk.data(), chunk_size, MPI_UNSIGNED_CHAR, image_out_full.data(), chunk_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    return true;     
}

bool LinearContrastStretchingMPI::PostProcessingImpl() 
{
  return true;
}

}  // namespace kutergin_v_linear_contrast_stretching
