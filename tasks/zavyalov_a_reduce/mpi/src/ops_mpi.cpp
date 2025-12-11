#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <memory>

#include "zavyalov_a_reduce/common/include/common.hpp"

namespace zavyalov_a_reduce {

void ZavyalovAReduceMPI::MyReduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype type, MPI_Op operation,
                                  int root, MPI_Comm comm) {
  std::get<1>(GetOutput()) = false;
  int world_size = 0;
  int world_rank = 0;

  MPI_Comm_size(comm, &world_size);
  MPI_Comm_rank(comm, &world_rank);

  if (operation == MPI_SUM) {
    if (type == MPI_INT) {
      // Используем shared_ptr с custom deleter для массива int
      std::shared_ptr<int[]> temp_buf(new int[count], [](int *p) { delete[] p; });
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;

      if (world_rank == 0) {
        std::shared_ptr<int[]> res_buf(new int[count], [](int *p) { delete[] p; });
        std::memcpy(res_buf.get(), sendbuf, count * sizeof(int));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
        }

        int sender_rank = 1;
        while (sender_rank < world_size) {
          MPI_Recv(temp_buf.get(), count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);

          for (int i = 0; i < count; i++) {
            res_buf[i] += temp_buf[i];
          }
          sender_rank *= 2;
        }
        if (root != 0) {
          MPI_Send(res_buf.get(), count, type, root, world_rank, comm);
        } else {
          std::memcpy(recvbuf, res_buf.get(), count * sizeof(int));
        }
        // Память автоматически освободится при выходе из блока

      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          std::shared_ptr<int[]> res_buf(new int[count], [](int *p) { delete[] p; });
          std::memcpy(res_buf.get(), sendbuf, count * sizeof(int));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            MPI_Recv(temp_buf.get(), count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] += temp_buf[i];
            }
            child_offset *= 2;
          }
          MPI_Send(res_buf.get(), count, type, world_rank - parent_offset, world_rank, comm);
          // Память автоматически освободится
        } else {
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }
      // Память temp_buf автоматически освободится

    } else if (type == MPI_FLOAT) {
      std::shared_ptr<float[]> temp_buf(new float[count], [](float *p) { delete[] p; });
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;

      if (world_rank == 0) {
        std::shared_ptr<float[]> res_buf(new float[count], [](float *p) { delete[] p; });
        std::memcpy(res_buf.get(), sendbuf, count * sizeof(float));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
        }

        int sender_rank = 1;
        while (sender_rank < world_size) {
          MPI_Recv(temp_buf.get(), count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);

          for (int i = 0; i < count; i++) {
            res_buf[i] += temp_buf[i];
          }
          sender_rank *= 2;
        }
        if (root != 0) {
          MPI_Send(res_buf.get(), count, type, root, world_rank, comm);
        } else {
          std::memcpy(recvbuf, res_buf.get(), count * sizeof(float));
        }

      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          std::shared_ptr<float[]> res_buf(new float[count], [](float *p) { delete[] p; });
          std::memcpy(res_buf.get(), sendbuf, count * sizeof(float));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            MPI_Recv(temp_buf.get(), count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] += temp_buf[i];
            }
            child_offset *= 2;
          }
          MPI_Send(res_buf.get(), count, type, world_rank - parent_offset, world_rank, comm);
        } else {
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }

    } else if (type == MPI_DOUBLE) {
      std::shared_ptr<double[]> temp_buf(new double[count], [](double *p) { delete[] p; });
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;

      if (world_rank == 0) {
        std::shared_ptr<double[]> res_buf(new double[count], [](double *p) { delete[] p; });
        std::memcpy(res_buf.get(), sendbuf, count * sizeof(double));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
        }

        int sender_rank = 1;
        while (sender_rank < world_size) {
          MPI_Recv(temp_buf.get(), count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);

          for (int i = 0; i < count; i++) {
            res_buf[i] += temp_buf[i];
          }
          sender_rank *= 2;
        }
        if (root != 0) {
          MPI_Send(res_buf.get(), count, type, root, world_rank, comm);
        } else {
          std::memcpy(recvbuf, res_buf.get(), count * sizeof(double));
        }

      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          std::shared_ptr<double[]> res_buf(new double[count], [](double *p) { delete[] p; });
          std::memcpy(res_buf.get(), sendbuf, count * sizeof(double));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            MPI_Recv(temp_buf.get(), count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] += temp_buf[i];
            }
            child_offset *= 2;
          }
          MPI_Send(res_buf.get(), count, type, world_rank - parent_offset, world_rank, comm);
        } else {
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }
    }
  } else if (operation == MPI_MIN) {
    if (type == MPI_INT) {
      std::shared_ptr<int[]> temp_buf(new int[count], [](int *p) { delete[] p; });
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;

      if (world_rank == 0) {
        std::shared_ptr<int[]> res_buf(new int[count], [](int *p) { delete[] p; });
        std::memcpy(res_buf.get(), sendbuf, count * sizeof(int));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
        }

        int sender_rank = 1;
        while (sender_rank < world_size) {
          MPI_Recv(temp_buf.get(), count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);

          for (int i = 0; i < count; i++) {
            res_buf[i] = std::min(res_buf[i], temp_buf[i]);
          }
          sender_rank *= 2;
        }
        if (root != 0) {
          MPI_Send(res_buf.get(), count, type, root, world_rank, comm);
        } else {
          std::memcpy(recvbuf, res_buf.get(), count * sizeof(int));
        }

      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          std::shared_ptr<int[]> res_buf(new int[count], [](int *p) { delete[] p; });
          std::memcpy(res_buf.get(), sendbuf, count * sizeof(int));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            MPI_Recv(temp_buf.get(), count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] = std::min(res_buf[i], temp_buf[i]);
            }
            child_offset *= 2;
          }
          MPI_Send(res_buf.get(), count, type, world_rank - parent_offset, world_rank, comm);
        } else {
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }

    } else if (type == MPI_FLOAT) {
      std::shared_ptr<float[]> temp_buf(new float[count], [](float *p) { delete[] p; });
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;

      if (world_rank == 0) {
        std::shared_ptr<float[]> res_buf(new float[count], [](float *p) { delete[] p; });
        std::memcpy(res_buf.get(), sendbuf, count * sizeof(float));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
        }

        int sender_rank = 1;
        while (sender_rank < world_size) {
          MPI_Recv(temp_buf.get(), count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);

          for (int i = 0; i < count; i++) {
            res_buf[i] = std::min(res_buf[i], temp_buf[i]);
          }
          sender_rank *= 2;
        }
        if (root != 0) {
          MPI_Send(res_buf.get(), count, type, root, world_rank, comm);
        } else {
          std::memcpy(recvbuf, res_buf.get(), count * sizeof(float));
        }

      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          std::shared_ptr<float[]> res_buf(new float[count], [](float *p) { delete[] p; });
          std::memcpy(res_buf.get(), sendbuf, count * sizeof(float));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            MPI_Recv(temp_buf.get(), count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] = std::min(res_buf[i], temp_buf[i]);
            }
            child_offset *= 2;
          }
          MPI_Send(res_buf.get(), count, type, world_rank - parent_offset, world_rank, comm);
        } else {
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }

    } else if (type == MPI_DOUBLE) {
      std::shared_ptr<double[]> temp_buf(new double[count], [](double *p) { delete[] p; });
      int log2floored = 0;
      int tmp_rank = world_rank;
      int parent_offset = 1;

      if (world_rank == 0) {
        std::shared_ptr<double[]> res_buf(new double[count], [](double *p) { delete[] p; });
        std::memcpy(res_buf.get(), sendbuf, count * sizeof(double));
        tmp_rank = world_size;
        while (tmp_rank > 1) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
        }

        int sender_rank = 1;
        while (sender_rank < world_size) {
          MPI_Recv(temp_buf.get(), count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);

          for (int i = 0; i < count; i++) {
            res_buf[i] = std::min(res_buf[i], temp_buf[i]);
          }
          sender_rank *= 2;
        }
        if (root != 0) {
          MPI_Send(res_buf.get(), count, type, root, world_rank, comm);
        } else {
          std::memcpy(recvbuf, res_buf.get(), count * sizeof(double));
        }

      } else {
        while (tmp_rank % 2 == 0) {
          log2floored++;
          tmp_rank >>= 1;  // tmp_rank /= 2
          parent_offset *= 2;
        }
        if (log2floored > 0) {
          std::shared_ptr<double[]> res_buf(new double[count], [](double *p) { delete[] p; });
          std::memcpy(res_buf.get(), sendbuf, count * sizeof(double));
          int child_offset = 1;
          for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); iter++) {
            MPI_Recv(temp_buf.get(), count, type, world_rank + child_offset, world_rank + child_offset, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
              res_buf[i] = std::min(res_buf[i], temp_buf[i]);
            }
            child_offset *= 2;
          }
          MPI_Send(res_buf.get(), count, type, world_rank - parent_offset, world_rank, comm);
        } else {
          MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
        }

        if (world_rank == root) {
          MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
      }
    }
  }
}

ZavyalovAReduceMPI::ZavyalovAReduceMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::get<0>(GetOutput()) = std::shared_ptr<void>(nullptr);
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

  // Исправление для shared_ptr
  std::shared_ptr<void> mem_ptr = std::get<3>(GetInput());
  res &= (mem_ptr != nullptr);

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
  std::shared_ptr<void> mem_ptr = std::get<3>(GetInput());
  void *mem = mem_ptr.get();
  int receiver_rank = std::get<4>(GetInput());

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int type_size = 0;
  MPI_Type_size(cur_type, &type_size);

  char *raw_result = new char[sz * type_size];
  std::shared_ptr<void> result_ptr(raw_result, [](void *p) { delete[] static_cast<char *>(p); });

  if (rank == receiver_rank) {
    MyReduce(mem, raw_result, static_cast<int>(sz), cur_type, operation, receiver_rank, MPI_COMM_WORLD);
    MPI_Bcast(raw_result, static_cast<int>(sz), cur_type, receiver_rank, MPI_COMM_WORLD);
  } else {
    MyReduce(mem, nullptr, static_cast<int>(sz), cur_type, operation, receiver_rank, MPI_COMM_WORLD);
    MPI_Bcast(raw_result, static_cast<int>(sz), cur_type, receiver_rank, MPI_COMM_WORLD);
  }

  std::get<0>(GetOutput()) = result_ptr;
  std::get<1>(GetOutput()) = false;  // false для MPI версии

  return true;
}

bool ZavyalovAReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_reduce
