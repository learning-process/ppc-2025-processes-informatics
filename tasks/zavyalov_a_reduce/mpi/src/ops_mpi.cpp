#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "zavyalov_a_reduce/common/include/common.hpp"

namespace zavyalov_a_reduce {

void ZavyalovAReduceMPI::my_reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype type, MPI_Op operation,
                                   int root, MPI_Comm comm) {
  int world_size;
  int world_rank;

  MPI_Comm_size(comm, &world_size);
  MPI_Comm_rank(comm, &world_rank);

  if (operation == MPI_SUM) {
    if (type == MPI_INT) {
      int *temp_buf = new int[count];
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;
      if (world_rank == 0) {
        int *res_buf = new int[count];
        memcpy(res_buf, sendbuf, count * sizeof(int));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
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
        } else {
          memcpy(recvbuf, res_buf, count * sizeof(int));
        }
        delete[] res_buf;

      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          int *res_buf = new int[count];
          memcpy(res_buf, sendbuf, count * sizeof(int));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            std::cout << "process " << world_rank << " is receiving from " << world_rank + child_offset << std::endl;
            MPI_Recv(temp_buf, count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] += temp_buf[i];
            }
            child_offset *= 2;
          }
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(res_buf, count, type, world_rank - parent_offset, world_rank, comm);
          delete[] res_buf;
        } else {
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }
      delete[] temp_buf;
    } else if (type == MPI_FLOAT) {
      float *temp_buf = new float[count];
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;
      if (world_rank == 0) {
        float *res_buf = new float[count];
        memcpy(res_buf, sendbuf, count * sizeof(float));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
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
        } else {
          memcpy(recvbuf, res_buf, count * sizeof(float));
        }
      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          float *res_buf = new float[count];
          memcpy(res_buf, sendbuf, count * sizeof(float));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            std::cout << "process " << world_rank << " is receiving from " << world_rank + child_offset << std::endl;
            MPI_Recv(temp_buf, count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] += temp_buf[i];
            }
            child_offset *= 2;
          }
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(res_buf, count, type, world_rank - parent_offset, world_rank, comm);
          delete[] res_buf;
        } else {
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }
      delete[] temp_buf;
    } else if (type == MPI_DOUBLE) {
      double *temp_buf = new double[count];
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;
      if (world_rank == 0) {
        double *res_buf = new double[count];
        memcpy(res_buf, sendbuf, count * sizeof(double));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
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
        } else {
          memcpy(recvbuf, res_buf, count * sizeof(double));
        }
      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          double *res_buf = new double[count];
          memcpy(res_buf, sendbuf, count * sizeof(double));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            std::cout << "process " << world_rank << " is receiving from " << world_rank + child_offset << std::endl;
            MPI_Recv(temp_buf, count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] += temp_buf[i];
            }
            child_offset *= 2;
          }
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(res_buf, count, type, world_rank - parent_offset, world_rank, comm);
          delete[] res_buf;
        } else {
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }
      delete[] temp_buf;
    }
  } else if (operation == MPI_MIN) {
    if (type == MPI_INT) {
      int *temp_buf = new int[count];
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;
      if (world_rank == 0) {
        int *res_buf = new int[count];
        memcpy(res_buf, sendbuf, count * sizeof(int));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
        }

        int sender_rank = 1;
        while (sender_rank < world_size) {
          std::cout << "process " << world_rank << " is receiving from " << sender_rank << std::endl;
          MPI_Recv(temp_buf, count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);

          for (int i = 0; i < count; i++) {
            res_buf[i] = std::min(res_buf[i], temp_buf[i]);
          }
          sender_rank *= 2;
        }
        if (root != 0) {
          std::cout << "process " << world_rank << " is sending to " << root << std::endl;
          MPI_Send(res_buf, count, type, root, world_rank, comm);
        } else {
          memcpy(recvbuf, res_buf, count * sizeof(int));
        }
        delete[] res_buf;

      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          int *res_buf = new int[count];
          memcpy(res_buf, sendbuf, count * sizeof(int));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            std::cout << "process " << world_rank << " is receiving from " << world_rank + child_offset << std::endl;
            MPI_Recv(temp_buf, count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] = std::min(res_buf[i], temp_buf[i]);
            }
            child_offset *= 2;
          }
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(res_buf, count, type, world_rank - parent_offset, world_rank, comm);
          delete[] res_buf;
        } else {
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }
      delete[] temp_buf;
    } else if (type == MPI_FLOAT) {
      float *temp_buf = new float[count];
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;
      if (world_rank == 0) {
        float *res_buf = new float[count];
        memcpy(res_buf, sendbuf, count * sizeof(float));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
        }

        int sender_rank = 1;
        while (sender_rank < world_size) {
          std::cout << "process " << world_rank << " is receiving from " << sender_rank << std::endl;
          MPI_Recv(temp_buf, count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);

          for (int i = 0; i < count; i++) {
            res_buf[i] = std::min(res_buf[i], temp_buf[i]);
          }
          sender_rank *= 2;
        }
        if (root != 0) {
          std::cout << "process " << world_rank << " is sending to " << root << std::endl;
          MPI_Send(res_buf, count, type, root, world_rank, comm);
          delete[] res_buf;
        } else {
          memcpy(recvbuf, res_buf, count * sizeof(float));
        }
      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          float *res_buf = new float[count];
          memcpy(res_buf, sendbuf, count * sizeof(float));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            std::cout << "process " << world_rank << " is receiving from " << world_rank + child_offset << std::endl;
            MPI_Recv(temp_buf, count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] = std::min(res_buf[i], temp_buf[i]);
            }
            child_offset *= 2;
          }
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(res_buf, count, type, world_rank - parent_offset, world_rank, comm);
          delete[] res_buf;
        } else {
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }
      delete[] temp_buf;
    } else if (type == MPI_DOUBLE) {
      double *temp_buf = new double[count];
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;
      if (world_rank == 0) {
        double *res_buf = new double[count];
        memcpy(res_buf, sendbuf, count * sizeof(double));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
        }

        int sender_rank = 1;
        while (sender_rank < world_size) {
          std::cout << "process " << world_rank << " is receiving from " << sender_rank << std::endl;
          MPI_Recv(temp_buf, count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);

          for (int i = 0; i < count; i++) {
            res_buf[i] = std::min(res_buf[i], temp_buf[i]);
          }
          sender_rank *= 2;
        }
        if (root != 0) {
          std::cout << "process " << world_rank << " is sending to " << root << std::endl;
          MPI_Send(res_buf, count, type, root, world_rank, comm);
          delete[] res_buf;
        } else {
          memcpy(recvbuf, res_buf, count * sizeof(double));
        }
      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          double *res_buf = new double[count];
          memcpy(res_buf, sendbuf, count * sizeof(double));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            std::cout << "process " << world_rank << " is receiving from " << world_rank + child_offset << std::endl;
            MPI_Recv(temp_buf, count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] = std::min(res_buf[i], temp_buf[i]);
            }
            child_offset *= 2;
          }
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(res_buf, count, type, world_rank - parent_offset, world_rank, comm);
          delete[] res_buf;
        } else {
          std::cout << "process " << world_rank << " is sending to " << world_rank - parent_offset << std::endl;
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }
      delete[] temp_buf;
    }
  }
}

ZavyalovAReduceMPI::ZavyalovAReduceMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::get<0>(GetOutput()) = nullptr;
}

bool ZavyalovAReduceMPI::ValidationImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (rank != 0) {
    return true;
  }
  bool res = true;
  MPI_Op operation = std::get<0>(GetInput());
  res &= (operation == MPI_SUM || operation == MPI_MIN);
  MPI_Datatype cur_type = std::get<1>(GetInput());
  res &= (cur_type == MPI_INT || cur_type == MPI_FLOAT || cur_type == MPI_DOUBLE);
  size_t sz = std::get<2>(GetInput());
  res &= sz > 0;
  void *mem = std::get<3>(GetInput());
  res &= (mem != nullptr);
  int receiver_rank = std::get<4>(GetInput());
  res &= (receiver_rank < world_size);
  return res;
}

bool ZavyalovAReduceMPI::PreProcessingImpl() {
  return true;
}

bool ZavyalovAReduceMPI::RunImpl() {
  MPI_Op operation = std::get<0>(GetInput());
  MPI_Datatype cur_type = std::get<1>(GetInput());
  size_t sz = std::get<2>(GetInput());
  void *mem = std::get<3>(GetInput());
  int receiver_rank = std::get<4>(GetInput());

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int type_size;
  MPI_Type_size(cur_type, &type_size);
  void *result_buf = new char[sz * type_size];
  /* if (cur_type == MPI_INT) {
    result_buf = new int[sz]; // не всегда int[], в общем случае T[]
  } else if (cur_type == MPI_FLOAT) {
    result_buf = new float[sz];
  } else if (cur_type == MPI_DOUBLE) {
    result_buf = new double[sz];
  } */
  if (rank == receiver_rank) {
    my_reduce(mem, result_buf, sz, cur_type, operation, receiver_rank, MPI_COMM_WORLD);
    MPI_Bcast(result_buf, sz, cur_type, receiver_rank, MPI_COMM_WORLD);
  } else {
    my_reduce(mem, nullptr, sz, cur_type, operation, receiver_rank, MPI_COMM_WORLD);
    MPI_Bcast(result_buf, sz, cur_type, receiver_rank, MPI_COMM_WORLD);
  }

  std::get<0>(GetOutput()) = result_buf;

  return true;
}

bool ZavyalovAReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_reduce
