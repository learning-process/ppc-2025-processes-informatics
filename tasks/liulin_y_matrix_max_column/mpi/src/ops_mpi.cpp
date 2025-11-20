#include <mpi.h>
#include <vector>
#include <algorithm>
#include "liulin_y_matrix_max_column/mpi/include/ops_mpi.hpp"
#include "util/include/util.hpp"

namespace liulin_y_matrix_max_column {

// Турнирный максимум для одного столбца
int LiulinYMatrixMaxColumnMPI::tournament_max(const std::vector<int>& column) {
    int size = column.size();
    std::vector<int> temp = column;
    while (size > 1) {
        int new_size = 0;
        for (int i = 0; i < size; i += 2) {
            if (i + 1 < size)
                temp[new_size] = std::max(temp[i], temp[i + 1]);
            else
                temp[new_size] = temp[i];
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
    if (in.empty() || in[0].empty()) return false;

    size_t cols = in[0].size();
    for (const auto& row : in)
        if (row.size() != cols) return false;

    return GetOutput().empty();
}

bool LiulinYMatrixMaxColumnMPI::PreProcessingImpl() {
    size_t cols = GetInput()[0].size();
    GetOutput().assign(cols, std::numeric_limits<int>::min());
    return true;
}

bool LiulinYMatrixMaxColumnMPI::RunImpl() {
    const auto& in = GetInput();
    auto& out = GetOutput();

    if (in.empty()) return false;

    int rows = static_cast<int>(in.size());
    int cols = static_cast<int>(in[0].size());

    MPI_Init(nullptr, nullptr);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Вычисляем, сколько столбцов на процесс
    std::vector<int> sendcounts(world_size), displs(world_size);
    int base = cols / world_size;
    int rem = cols % world_size;
    for (int i = 0; i < world_size; ++i) {
        int local_cols = base + (i < rem ? 1 : 0);
        sendcounts[i] = local_cols * rows;
        displs[i] = (i == 0 ? 0 : displs[i - 1] + sendcounts[i - 1]);
    }

    // Flatten column-major
    std::vector<int> flat_matrix;
    if (world_rank == 0) {
        flat_matrix.resize(rows * cols);
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                flat_matrix[j * rows + i] = in[i][j];
    }

    int local_cols = sendcounts[world_rank] / rows;
    std::vector<int> local_flat(local_cols * rows);

    MPI_Scatterv(flat_matrix.data(), sendcounts.data(), displs.data(), MPI_INT,
                 local_flat.data(), local_flat.size(), MPI_INT, 0, MPI_COMM_WORLD);

    // Каждый процесс считает максимум по своим столбцам
    std::vector<int> local_max(local_cols);
    for (int j = 0; j < local_cols; ++j) {
        std::vector<int> col(rows);
        for (int i = 0; i < rows; ++i)
            col[i] = local_flat[j * rows + i];
        local_max[j] = tournament_max(col);
    }

    // Собираем результаты обратно на root
    std::vector<int> recvcounts(world_size), displs_gather(world_size);
    for (int i = 0; i < world_size; ++i)
        recvcounts[i] = sendcounts[i] / rows;
    for (int i = 0; i < world_size; ++i)
        displs_gather[i] = (i == 0 ? 0 : displs_gather[i - 1] + recvcounts[i - 1]);

    MPI_Gatherv(local_max.data(), local_max.size(), MPI_INT,
                out.data(), recvcounts.data(), displs_gather.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Finalize();
    return true;
}

bool LiulinYMatrixMaxColumnMPI::PostProcessingImpl() {
    return true;
}

} // namespace liulin_y_matrix_max_column
