#include "ovsyannikov_n_star/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include "ovsyannikov_n_star/common/include/common.hpp"

namespace ovsyannikov_n_star {

OvsyannikovNStarMPI::OvsyannikovNStarMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput() = 0;
}

bool OvsyannikovNStarMPI::ValidationImpl() {
    // Проверка, что входных данных ровно 3 (src, dst, val)
    return GetInput().size() == 3;
}

bool OvsyannikovNStarMPI::PreProcessingImpl() {
    GetOutput() = 0;
    return true;
}

bool OvsyannikovNStarMPI::RunImpl() {
    int is_initialized = 0;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
        GetOutput() = GetInput()[2];
        return true;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int src = GetInput()[0];
    int dst = GetInput()[1];
    int val = GetInput()[2];

    // Если данные некорректны для текущего количества процессов
    if (size <= 1 || src >= size || dst >= size || src < 0 || dst < 0) {
        if (rank == 0) GetOutput() = val;
        return true; 
    }

    int res = 0;
    bool is_transit = (src != 0 && dst != 0);

    if (rank == src) {
        if (src == dst) {
            res = val;
        } else {
            int next_hop = is_transit ? 0 : dst;
            MPI_Send(&val, 1, MPI_INT, next_hop, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0 && is_transit) {
        int tmp;
        MPI_Recv(&tmp, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&tmp, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }

    if (rank == dst && src != dst) {
        int source_hop = is_transit ? 0 : src;
        MPI_Recv(&res, 1, MPI_INT, source_hop, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    GetOutput() = res;
    // Рассылаем результат от dst всем остальным
    MPI_Bcast(&GetOutput(), 1, MPI_INT, dst, MPI_COMM_WORLD);

    return true;
}

bool OvsyannikovNStarMPI::PostProcessingImpl() {
    return true;
}

}  // namespace ovsyannikov_n_star