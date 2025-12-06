#include "lukin_i_cannon_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <tuple>
#include <vector>

#include "lukin_i_cannon_algorithm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace lukin_i_cannon_algorithm {

LukinICannonAlgorithmMPI::LukinICannonAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
  GetOutput() = OutType();
}

bool LukinICannonAlgorithmMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    int rsizeA = std::get<0>(GetInput()).size();
    int rsizeB = std::get<1>(GetInput()).size();
    size_ = std::get<2>(GetInput());
    return (rsizeA > 0) && (rsizeB > 0) && (rsizeA == size_ * size_) && (rsizeA == rsizeB);
  }
  return true;
}

bool LukinICannonAlgorithmMPI::PreProcessingImpl() {
  return true;
}

bool LukinICannonAlgorithmMPI::RunImpl() {
  int global_rank = -1;
  int proc_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  if (global_rank == 0) {
    std::cerr << "DEBUG: rank=" << global_rank << ", proc_count=" << proc_count << std::endl;
  }

  if (proc_count < 4) {
    std::cerr << "DEBUG: Using sequential version (proc_count=" << proc_count << " < 4)" << std::endl;
    // ...
  } else {
    std::cerr << "DEBUG: Using Cannon algorithm" << std::endl;
    // ...
  }

  MPI_Bcast(&size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // случай, если 1 на 1 решетка процессов - просто seq версия
  if (proc_count < 4) {
    std::vector<double> C(size_ * size_);
    if (global_rank == 0) {
      double *A = std::get<0>(GetInput()).data();
      double *B = std::get<1>(GetInput()).data();
      mul_n_sum(A, B, C.data(), size_);
    }
    MPI_Bcast(C.data(), size_ * size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    GetOutput() = std::move(C);
    return true;
  }
  std::cerr << "I'm here!!!" << std::endl;
  // для процессов, напрямую учавствующих в вычислениях, создается другой коммуникатор
  int grid_size = static_cast<int>(std::floor(std::sqrt(proc_count)));
  int working_proc_count = grid_size * grid_size;
  MPI_Comm MPI_COMM_CANNON;
  int color = (global_rank < working_proc_count) ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, global_rank, &MPI_COMM_CANNON);

  if (MPI_COMM_CANNON != MPI_COMM_NULL) {
    int cannon_rank = -1;
    MPI_Comm_rank(MPI_COMM_CANNON, &cannon_rank);

    int block_size = size_ / grid_size;
    int block_elems = block_size * block_size;
    std::vector<double> A_block(block_elems);
    std::vector<double> B_block(block_elems);
    std::vector<double> C_block(block_elems, 0);

    std::vector<double> A_blocks;
    std::vector<double> B_blocks;

    // ручная упаковка
    if (cannon_rank == 0) {
      double *A = std::get<0>(GetInput()).data();
      double *B = std::get<1>(GetInput()).data();

      A_blocks.resize(working_proc_count * block_elems);
      B_blocks.resize(working_proc_count * block_elems);

      for (int proc = 0; proc < working_proc_count; proc++) {
        int proc_i = proc / grid_size;
        int proc_j = proc % grid_size;
        int buf_offset = proc * block_elems;

        for (int i = 0; i < block_size; i++) {
          for (int j = 0; j < block_size; j++) {
            int global_i = proc_i * block_size + i;
            int global_j = proc_j * block_size + j;
            int global_idx = global_i * size_ + global_j;
            int buf_idx = buf_offset + i * block_size + j;

            A_blocks[buf_idx] = A[global_idx];
            B_blocks[buf_idx] = B[global_idx];
          }
        }
      }
    }

    MPI_Scatter(A_blocks.data(), block_elems, MPI_DOUBLE, A_block.data(), block_elems, MPI_DOUBLE, 0, MPI_COMM_CANNON);

    MPI_Scatter(B_blocks.data(), block_elems, MPI_DOUBLE, B_block.data(), block_elems, MPI_DOUBLE, 0, MPI_COMM_CANNON);

    int row = cannon_rank / grid_size;
    int col = cannon_rank % grid_size;

    // начальный сдвиг
    int left = (row * grid_size) + ((col - row + grid_size) % grid_size);
    int right = (row * grid_size) + ((col + row) % grid_size);

    MPI_Sendrecv_replace(A_block.data(), block_elems, MPI_DOUBLE, left, 0, right, 0, MPI_COMM_CANNON,
                         MPI_STATUS_IGNORE);

    int up = ((row - col + grid_size) % grid_size) * grid_size + col;
    int down = ((row + col) % grid_size) * grid_size + col;

    MPI_Sendrecv_replace(B_block.data(), block_elems, MPI_DOUBLE, up, 0, down, 0, MPI_COMM_CANNON, MPI_STATUS_IGNORE);

    // цикл умножения и сдвига
    for (int iter = 0; iter < grid_size; iter++) {
      mul_n_sum(A_block.data(), B_block.data(), C_block.data(), block_size);

      if (iter < grid_size - 1) {
        left = row * grid_size + ((col - 1 + grid_size) % grid_size);
        right = row * grid_size + ((col + 1) % grid_size);

        MPI_Sendrecv_replace(A_block.data(), block_elems, MPI_DOUBLE, left, 0, right, 0, MPI_COMM_CANNON,
                             MPI_STATUS_IGNORE);

        up = ((row - 1 + grid_size) % grid_size) * grid_size + col;
        down = ((row + 1) % grid_size) * grid_size + col;

        MPI_Sendrecv_replace(B_block.data(), block_elems, MPI_DOUBLE, up, 0, down, 0, MPI_COMM_CANNON,
                             MPI_STATUS_IGNORE);
      }
    }

    // упаковка данных в результирующую
    std::vector<double> C_blocks(size_ * size_);
    MPI_Gather(C_block.data(), block_elems, MPI_DOUBLE, C_blocks.data(), block_elems, MPI_DOUBLE, 0, MPI_COMM_CANNON);

    std::vector<double> C(size_ * size_);
    if (cannon_rank == 0) {
      for (int proc = 0; proc < working_proc_count; proc++) {
        int proc_i = proc / grid_size;
        int proc_j = proc % grid_size;
        int buf_offset = proc * block_elems;

        for (int i = 0; i < block_size; i++) {
          for (int j = 0; j < block_size; j++) {
            int global_i = proc_i * block_size + i;
            int global_j = proc_j * block_size + j;
            int global_idx = global_i * size_ + global_j;
            int buf_idx = buf_offset + i * block_size + j;

            C[global_idx] = C_blocks[buf_idx];
          }
        }
      }
    }

    MPI_Bcast(C.data(), size_ * size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    GetOutput() = std::move(C);

    MPI_Comm_free(&MPI_COMM_CANNON);
  } else {
    std::vector<double> C(size_ * size_);
    MPI_Bcast(C.data(), size_ * size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    GetOutput() = std::move(C);
  }

  return true;
}

bool LukinICannonAlgorithmMPI::PostProcessingImpl() {
  return true;
}

void LukinICannonAlgorithmMPI::mul_n_sum(double *A, double *B, double *C, const int size) {
  for (int i = 0; i < size; i++) {
    for (int k = 0; k < size; k++) {
      double fixed = A[i * size + k];
      for (int j = 0; j < size; j++) {
        C[i * size + j] += fixed * B[k * size + j];
      }
    }
  }
}

}  // namespace lukin_i_cannon_algorithm
