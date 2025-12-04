#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "zavyalov_a_reduce/common/include/common.hpp"

namespace zavyalov_a_reduce {


void ZavyalovAReduceMPI::my_reduce(const void *sendbuf, void* recvbuf, int count, MPI_Datatype type, MPI_Op operation, int root, MPI_Comm comm) {
  int world_size;
  int world_rank;

  MPI_Comm_size(comm, &world_size);
  MPI_Comm_rank(comm, &world_rank);

  if (type == MPI_INT) {
    if (operation == MPI_SUM) {
      int* temp_buf = new int[count];
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;
      if (world_rank == 0) {
        int* res_buf = new int[count];
        if (root != 0) {
          std::memcpy(res_buf, sendbuf, count * sizeof(int));
        }
        else {
          for (int i = 0; i < count; i++) {
            res_buf[i] = 0;
          }
        }
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1; // tmp_rank /= 2
        }

        int sender_rank = 1;
        while (sender_rank < world_size) {
            std::cout << "process " << world_rank << " is receiving from " << sender_rank << std::endl;
            MPI_Recv(temp_buf, count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);

            for (int i = 0; i < count; i++) {
                res_buf[i] += temp_buf[i];
            }
            sender_rank *= 2;
        }
        if (root != 0) {
            std::cout << "process " << world_rank << " is sending to " << root << std::endl;
            MPI_Send(res_buf, count, type, root, world_rank, comm);
            delete[] res_buf;
        }
        else {
            std::memcpy(recvbuf, res_buf, count * sizeof(int));
        }
      }
      else {
        while (tmp_rank % 2 == 0) {
            log2floored++;
            tmp_rank >>= 1; // tmp_rank /= 2
            parent_offset *= 2;
        }
        int* res_buf;
        if (log2floored > 0) {
            res_buf = new int[count];
            std::memcpy(res_buf, sendbuf, count * sizeof(int));
            int child_offset = 1;
            for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
                std::cout << "process " << world_rank << " is receiving from " << world_rank + child_offset << std::endl;
                MPI_Recv(temp_buf, count, type, world_rank + child_offset, world_rank + child_offset, comm, MPI_STATUS_IGNORE);
                for (int i = 0; i < count; i++) {
                    res_buf[i] += temp_buf[i];
                }
                child_offset *= 2;
            }
            std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
            MPI_Send(res_buf, count, type, world_rank - parent_offset, world_rank, comm);
            delete[] res_buf;
        }
        else {
            std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
            MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }
      }
    }
  }
}


ZavyalovAReduceMPI::ZavyalovAReduceMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
  GetOutput() = nullptr;
}

bool ZavyalovAReduceMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  return true;
  // return (!std::get<0>(GetInput()).empty()) && (std::get<0>(GetInput()).size() == std::get<1>(GetInput()).size());
}

bool ZavyalovAReduceMPI::PreProcessingImpl() {
  return true;
}

bool ZavyalovAReduceMPI::RunImpl() {
  return true;
  /*
  const double *left_data = nullptr;
  const double *right_data = nullptr;

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int vector_size = 0;

  if (rank == 0) {
    GetOutput() = 0.0;
    const auto &input = GetInput();
    if (!std::get<0>(input).empty()) {  // it does not compile in ubuntu without this line
      left_data = std::get<0>(input).data();
    }
    if (!std::get<1>(input).empty()) {  // it does not compile in ubuntu without this line
      right_data = std::get<1>(input).data();
    }
    vector_size = static_cast<int>(std::get<0>(input).size());
  }

  MPI_Bcast(&vector_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);

  int blocksize = vector_size / world_size;
  int elements_left = vector_size - (world_size * blocksize);
  int elements_processed = 0;

  for (int i = 0; i < world_size; i++) {
    sendcounts[i] = blocksize + (i < elements_left ? 1 : 0);
    displs[i] = elements_processed;
    elements_processed += sendcounts[i];
  }

  int elements_count = sendcounts[rank];
  std::vector<double> local_left(elements_count);
  std::vector<double> local_right(elements_count);

  MPI_Scatterv(left_data, sendcounts.data(), displs.data(), MPI_DOUBLE, local_left.data(), elements_count, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
  MPI_Scatterv(right_data, sendcounts.data(), displs.data(), MPI_DOUBLE, local_right.data(), elements_count, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

  double cur_res = 0.0;
  for (int i = 0; i < elements_count; i++) {
    cur_res += local_left[i] * local_right[i];
  }

  double glob_res = 0.0;
  MPI_Allreduce(&cur_res, &glob_res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = glob_res;

  return true;
  */
}

bool ZavyalovAReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_reduce
