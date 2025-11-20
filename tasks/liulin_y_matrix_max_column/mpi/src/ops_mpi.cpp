#include <mpi.h>
#include <vector>
#include <algorithm>
#include <limits>
#include "liulin_y_matrix_max_column/mpi/include/ops_mpi.hpp"
#include "util/include/util.hpp"

namespace liulin_y_matrix_max_column {

int LiulinYMatrixMaxColumnMPI::tournament_max(const std::vector<int>& column) {
    if (column.empty()) return std::numeric_limits<int>::min();
    
    int size = column.size();
    std::vector<int> temp = column;
    
    while (size > 1) {
        int new_size = 0;
        for (int i = 0; i < size; i += 2) {
            if (i + 1 < size) {
                temp[new_size] = std::max(temp[i], temp[i + 1]);
            } else {
                temp[new_size] = temp[i];
            }
            new_size++;
        }
        size = new_size;
    }
    return temp[0];
}

LiulinYMatrixMaxColumnMPI::LiulinYMatrixMaxColumnMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput().clear();
}

bool LiulinYMatrixMaxColumnMPI::ValidationImpl() {
    const auto& in = GetInput();
    if (in.empty()) return false;

    size_t cols = in[0].size();
    for (const auto& row : in) {
        if (row.size() != cols) return false;
    }

    return true;
}

bool LiulinYMatrixMaxColumnMPI::PreProcessingImpl() {
    // Инициализация выхода будет в RunImpl после определения cols
    return true;
}

bool LiulinYMatrixMaxColumnMPI::RunImpl() {
    const auto& in = GetInput();
    auto& out = GetOutput();

    int world_size = 0, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Определяем размеры матрицы
    int rows = 0, cols = 0;
    if (world_rank == 0) {
        if (!in.empty() && !in[0].empty()) {
            rows = static_cast<int>(in.size());
            cols = static_cast<int>(in[0].size());
        }
    }

    // Рассылаем размеры всем процессам
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Проверяем валидность размеров
    if (rows <= 0 || cols <= 0) {
        out.clear();
        return true;
    }

    // Инициализируем выходной вектор на всех процессах
    out.resize(cols, std::numeric_limits<int>::min());

    // Распределение столбцов по процессам
    std::vector<int> sendcounts(world_size, 0);
    std::vector<int> displs(world_size, 0);
    
    int base_cols = cols / world_size;
    int remainder = cols % world_size;
    
    for (int i = 0; i < world_size; ++i) {
        int local_cols = base_cols + (i < remainder ? 1 : 0);
        sendcounts[i] = local_cols * rows;
        if (i > 0) {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }
    }

    // Подготовка данных для рассылки (только на root)
    std::vector<int> flat_matrix;
    if (world_rank == 0) {
        flat_matrix.resize(rows * cols);
        // Column-major order
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                flat_matrix[j * rows + i] = in[i][j];
            }
        }
    }

    // Определяем локальные размеры
    int local_cols = sendcounts[world_rank] / rows;
    int local_elements = local_cols * rows;
    std::vector<int> local_data(local_elements);

    // Рассылаем данные
    MPI_Scatterv(
        world_rank == 0 ? flat_matrix.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_INT,
        local_data.data(),
        local_elements,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Вычисляем локальные максимумы
    std::vector<int> local_maxes(local_cols, std::numeric_limits<int>::min());
    for (int j = 0; j < local_cols; ++j) {
        std::vector<int> column(rows);
        for (int i = 0; i < rows; ++i) {
            column[i] = local_data[j * rows + i];
        }
        local_maxes[j] = tournament_max(column);
    }

    // Подготавливаем параметры для сбора
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs_gather(world_size, 0);
    
    for (int i = 0; i < world_size; ++i) {
        recvcounts[i] = sendcounts[i] / rows; // количество столбцов
        if (i > 0) {
            displs_gather[i] = displs_gather[i-1] + recvcounts[i-1];
        }
    }

    // Собираем результаты
    MPI_Gatherv(
        local_maxes.data(),
        local_cols,
        MPI_INT,
        out.data(),
        recvcounts.data(),
        displs_gather.data(),
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Синхронизируем результаты на всех процессах
    MPI_Bcast(out.data(), cols, MPI_INT, 0, MPI_COMM_WORLD);

    return true;
}

bool LiulinYMatrixMaxColumnMPI::PostProcessingImpl() {
    return true;
}

} // namespace liulin_y_matrix_max_column