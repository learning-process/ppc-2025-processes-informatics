#include "liulin_y_matrix_max_column/mpi/include/ops_mpi.hpp"  

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "liulin_y_matrix_max_column/common/include/common.hpp"

namespace liulin_y_matrix_max_column {

// Реализация статического метода класса
int LiulinYMatrixMaxColumnMPI::TournamentMax(const std::vector<int> &column) {
    if (column.empty()) {
        return std::numeric_limits<int>::min();
    }

    int size = static_cast<int>(column.size());
    std::vector<int> temp = column;

    while (size > 1) {
        int new_size = 0;
        for (int i = 0; i < size; i += 2) {
            if (i + 1 < size) {
                temp[new_size] = std::max(temp[i], temp[i + 1]);
            } else {
                temp[new_size] = temp[i];
            }
            ++new_size;
        }
        size = new_size;
    }
    return temp[0];
}

// Конструктор
LiulinYMatrixMaxColumnMPI::LiulinYMatrixMaxColumnMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput().clear();
}

// Валидация
bool LiulinYMatrixMaxColumnMPI::ValidationImpl() {
    const auto &in = GetInput();
    if (in.empty()) {
        return false;
    }

    size_t cols = in[0].size();
    for (const auto &row : in) {
        if (row.size() != cols) {
            return false;
        }
    }
    return true;
}

// Предобработка
bool LiulinYMatrixMaxColumnMPI::PreProcessingImpl() {
    return true;
}

// Основной метод Run
bool LiulinYMatrixMaxColumnMPI::RunImpl() {
    const auto &in = GetInput();
    auto &out = GetOutput();

    int world_size = 0, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int rows = 0, cols = 0;
    if (world_rank == 0 && !in.empty() && !in[0].empty()) {
        rows = static_cast<int>(in.size());
        cols = static_cast<int>(in[0].size());
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rows <= 0 || cols <= 0) {
        out.clear();
        return true;
    }

    out.resize(cols, std::numeric_limits<int>::min());

    std::vector<int> sendcounts(world_size, 0);
    std::vector<int> displs(world_size, 0);

    int base_cols = cols / world_size;
    int remainder = cols % world_size;
    
    for (int it = 0; it < world_size; ++it) {
        int local_cols = 0;
        local_cols = base_cols + (it < remainder ? 1 : 0);
        sendcounts[it] = local_cols * rows;
        if (it > 0) {
            displs[it] = displs[it - 1] + sendcounts[it - 1];
        }
    }

    std::vector<int> flat_matrix;
    if (world_rank == 0) {
        flat_matrix.resize(rows * cols);
        for (int jt = 0; jt < cols; ++jt) {
            for (int it = 0; it < rows; ++it) {
                flat_matrix[jt * rows + it] = in[it][jt];
            }
        }
    }

    int local_cols = sendcounts[world_rank] / rows;
    int local_elements = local_cols * rows;
    std::vector<int> local_data(local_elements);

    MPI_Scatterv(world_rank == 0 ? flat_matrix.data() : nullptr,
                 sendcounts.data(),
                 displs.data(),
                 MPI_INT,
                 local_data.data(),
                 local_elements,
                 MPI_INT,
                 0,
                 MPI_COMM_WORLD);

    std::vector<int> local_maxes(local_cols, std::numeric_limits<int>::min());
    for (int jt = 0; jt < local_cols; ++jt) {
        std::vector<int> column(rows);
        for (int it = 0; it < rows; ++it) {
            column[it] = local_data[jt * rows + it];
        }
        local_maxes[jt] = TournamentMax(column);
    }

    std::vector<int> recvcounts(world_size);
    std::vector<int> displs_gather(world_size, 0);

    for (int it = 0; it < world_size; ++it) {
        recvcounts[it] = sendcounts[it] / rows;
        if (it > 0) {
            displs_gather[it] = displs_gather[it - 1] + recvcounts[it - 1];
        }
    }

    MPI_Gatherv(local_maxes.data(),
                local_cols,
                MPI_INT,
                out.data(),
                recvcounts.data(),
                displs_gather.data(),
                MPI_INT,
                0,
                MPI_COMM_WORLD);

    MPI_Bcast(out.data(), cols, MPI_INT, 0, MPI_COMM_WORLD);

    return true;
}

// Постобработка
bool LiulinYMatrixMaxColumnMPI::PostProcessingImpl() {
    return true;
}

}  // namespace liulin_y_matrix_max_column
