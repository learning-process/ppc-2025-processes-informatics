#include "smyshlaev_a_str_order_check/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <string>

#include "smyshlaev_a_str_order_check/common/include/common.hpp"

namespace smyshlaev_a_str_order_check {

SmyshlaevAStrOrderCheckMPI::SmyshlaevAStrOrderCheckMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput() = 0;
}

bool SmyshlaevAStrOrderCheckMPI::ValidationImpl() {
    return true;
}

bool SmyshlaevAStrOrderCheckMPI::PreProcessingImpl() {
    return true;
}

bool SmyshlaevAStrOrderCheckMPI::RunImpl() {

    const auto& input_data = GetInput();
    const std::string &str1 = input_data.first;
    const std::string &str2 = input_data.second;

    if (str1.empty() && str2.empty()) {
        GetOutput() = 0;
        return true;
    }

    int rank = 0;
    size_t size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_result = 0;
    int global_result = 0;
    
    size_t min_len = std::min(str1.length(), str2.length());

    if (rank == 0) {
        if (min_len == 0) {
            if (str1.length() < str2.length()) {
                global_result = -1;
            } else if (str1.length() > str2.length()) {
                global_result = 1;
            } else {
                global_result = 0;
            }
        }

        else {
            size_t chunk_size = min_len / size;
            size_t remainder = min_len % size;
            
            for (size_t i = 0; i < size; ++i) {
                size_t start_idx = i * chunk_size + std::min((size_t)i, remainder);
                size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);

                if (i != 0) {
                    MPI_Send(&start_idx, 1, MPI_UNSIGNED_LONG, i, 1, MPI_COMM_WORLD);
                    MPI_Send(&current_chunk_size, 1, MPI_UNSIGNED_LONG, i, 2, MPI_COMM_WORLD);
                    MPI_Send(str1.c_str() + start_idx, current_chunk_size, MPI_CHAR, i, 3, MPI_COMM_WORLD);
                    MPI_Send(str2.c_str() + start_idx, current_chunk_size, MPI_CHAR, i, 4, MPI_COMM_WORLD);
                } else {
                    local_result = 0;
                    for (size_t k = 0; k < current_chunk_size; ++k) {
                        if (str1[start_idx + k] < str2[start_idx + k]) {
                            local_result = -1;
                            break;
                        }
                        if (str1[start_idx + k] > str2[start_idx + k]) {
                            local_result = 1;
                            break;
                        }
                    }
                }
            }

            global_result = local_result;

            for (size_t i = 1; i < size; ++i) {
                int worker_result;
                MPI_Recv(&worker_result, 1, MPI_INT, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (global_result == 0 && worker_result != 0) {
                    global_result = worker_result;
                }
            }

            if (global_result == 0) {
                if (str1.length() < str2.length()) {
                    global_result = -1;
                } else if (str1.length() > str2.length()) {
                    global_result = 1;
                }
            }
        }

        GetOutput() = global_result;

    } else {
        size_t start_idx = 0;
        size_t current_chunk_size = 0;

        MPI_Recv(&start_idx, 1, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&current_chunk_size, 1, MPI_UNSIGNED_LONG, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (current_chunk_size > 0) {
            std::string sub_str1(current_chunk_size, '\0');
            std::string sub_str2(current_chunk_size, '\0');

            MPI_Recv(&sub_str1[0], current_chunk_size, MPI_CHAR, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&sub_str2[0], current_chunk_size, MPI_CHAR, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (size_t k = 0; k < current_chunk_size; ++k) {
                if (sub_str1[k] < sub_str2[k]) {
                    local_result = -1;
                    break;
                }
                if (sub_str1[k] > sub_str2[k]) {
                    local_result = 1;
                    break;
                }
            }
        }

        MPI_Send(&local_result, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return true;
}

bool SmyshlaevAStrOrderCheckMPI::PostProcessingImpl() {
    return true;
}

}  // namespace smyshlaev_a_str_order_check