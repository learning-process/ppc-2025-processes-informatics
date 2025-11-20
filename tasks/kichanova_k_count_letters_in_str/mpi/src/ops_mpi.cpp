#include "kichanova_k_count_letters_in_str/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cctype>
#include <string>

#include "kichanova_k_count_letters_in_str/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kichanova_k_count_letters_in_str {

KichanovaKCountLettersInStrMPI::KichanovaKCountLettersInStrMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KichanovaKCountLettersInStrMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool KichanovaKCountLettersInStrMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KichanovaKCountLettersInStrMPI::RunImpl() {
    auto input_str = GetInput();
    if (input_str.empty()) {
        return false;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_length = input_str.length();
    int chunk_size = total_length / size;
    
    int start_index = rank * chunk_size;
    int end_index = (rank == size - 1) ? total_length : start_index + chunk_size;

    int local_count = 0;
    for (int i = start_index; i < end_index; i++) {
        if (std::isalpha(static_cast<unsigned char>(input_str[i]))) {
            local_count++;
        }
    }

    int global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Bcast(&global_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    GetOutput() = global_count;

    return true;
}

bool KichanovaKCountLettersInStrMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace kichanova_k_count_letters_in_str
