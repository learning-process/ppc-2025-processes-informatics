#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "zavyalov_a_reduce/common/include/common.hpp"

namespace zavyalov_a_reduce {

namespace {  // анонимное пространство имён для внутренних шаблонов

template <typename T>
void ReduceSumImpl(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm, MPI_Datatype type) {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(comm, &world_size);
  MPI_Comm_rank(comm, &world_rank);

  std::vector<T> temp_buf(count);

  int log2floored = 0;
  int tmp_rank = world_rank;
  int parent_offset = 1;

  if (world_rank == 0) {
    std::vector<T> res_buf(count);
    std::memcpy(res_buf.data(), sendbuf, static_cast<std::size_t>(count) * sizeof(T));

    tmp_rank = world_size;
    while (tmp_rank > 1) {
      ++log2floored;
      tmp_rank >>= 1;
    }

    int sender_rank = 1;
    while (sender_rank < world_size) {
      MPI_Recv(temp_buf.data(), count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);
      for (int i = 0; i < count; ++i) {
        res_buf[i] += temp_buf[i];
      }
      sender_rank *= 2;
    }

    if (root != 0) {
      MPI_Send(res_buf.data(), count, type, root, world_rank, comm);
    } else {
      std::memcpy(recvbuf, res_buf.data(), static_cast<std::size_t>(count) * sizeof(T));
    }

  } else {
    while ((tmp_rank % 2) == 0) {
      ++log2floored;
      tmp_rank >>= 1;
      parent_offset *= 2;
    }

    if (log2floored > 0) {
      std::vector<T> res_buf(count);
      std::memcpy(res_buf.data(), sendbuf, static_cast<std::size_t>(count) * sizeof(T));

      int child_offset = 1;
      for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); ++iter) {
        MPI_Recv(temp_buf.data(), count, type, world_rank + child_offset, world_rank + child_offset, comm,
                 MPI_STATUS_IGNORE);
        for (int i = 0; i < count; ++i) {
          res_buf[i] += temp_buf[i];
        }
        child_offset *= 2;
      }

      MPI_Send(res_buf.data(), count, type, world_rank - parent_offset, world_rank, comm);
    } else {
      MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
    }

    if (world_rank == root) {
      MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
    }
  }
}

template <typename T>
void ReduceMinImpl(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm, MPI_Datatype type) {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(comm, &world_size);
  MPI_Comm_rank(comm, &world_rank);

  std::vector<T> temp_buf(count);

  int log2floored = 0;
  int tmp_rank = world_rank;
  int parent_offset = 1;

  if (world_rank == 0) {
    std::vector<T> res_buf(count);
    std::memcpy(res_buf.data(), sendbuf, static_cast<std::size_t>(count) * sizeof(T));

    tmp_rank = world_size;
    while (tmp_rank > 1) {
      ++log2floored;
      tmp_rank >>= 1;
    }

    int sender_rank = 1;
    while (sender_rank < world_size) {
      MPI_Recv(temp_buf.data(), count, type, sender_rank, sender_rank, comm, MPI_STATUS_IGNORE);
      for (int i = 0; i < count; ++i) {
        res_buf[i] = std::min(res_buf[i], temp_buf[i]);
      }
      sender_rank *= 2;
    }

    if (root != 0) {
      MPI_Send(res_buf.data(), count, type, root, world_rank, comm);
    } else {
      std::memcpy(recvbuf, res_buf.data(), static_cast<std::size_t>(count) * sizeof(T));
    }

  } else {
    while ((tmp_rank % 2) == 0) {
      ++log2floored;
      tmp_rank >>= 1;
      parent_offset *= 2;
    }

    if (log2floored > 0) {
      std::vector<T> res_buf(count);
      std::memcpy(res_buf.data(), sendbuf, static_cast<std::size_t>(count) * sizeof(T));

      int child_offset = 1;
      for (int iter = 1; (iter <= log2floored) && ((world_rank + child_offset) < world_size); ++iter) {
        MPI_Recv(temp_buf.data(), count, type, world_rank + child_offset, world_rank + child_offset, comm,
                 MPI_STATUS_IGNORE);
        for (int i = 0; i < count; ++i) {
          res_buf[i] = std::min(res_buf[i], temp_buf[i]);
        }
        child_offset *= 2;
      }

      MPI_Send(res_buf.data(), count, type, world_rank - parent_offset, world_rank, comm);
    } else {
      MPI_Send(sendbuf, count, type, world_rank - parent_offset, world_rank, comm);
    }

    if (world_rank == root) {
      MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
    }
  }
}

}  // namespace

// Тонкие обёртки, чтобы сохранить прежний интерфейс класса
void ZavyalovAReduceMPI::ReduceSumInt(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm) {
  ReduceSumImpl<int>(sendbuf, recvbuf, count, root, comm, MPI_INT);
}
void ZavyalovAReduceMPI::ReduceSumFloat(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm) {
  ReduceSumImpl<float>(sendbuf, recvbuf, count, root, comm, MPI_FLOAT);
}
void ZavyalovAReduceMPI::ReduceSumDouble(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm) {
  ReduceSumImpl<double>(sendbuf, recvbuf, count, root, comm, MPI_DOUBLE);
}

void ZavyalovAReduceMPI::ReduceMinInt(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm) {
  ReduceMinImpl<int>(sendbuf, recvbuf, count, root, comm, MPI_INT);
}
void ZavyalovAReduceMPI::ReduceMinFloat(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm) {
  ReduceMinImpl<float>(sendbuf, recvbuf, count, root, comm, MPI_FLOAT);
}
void ZavyalovAReduceMPI::ReduceMinDouble(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm) {
  ReduceMinImpl<double>(sendbuf, recvbuf, count, root, comm, MPI_DOUBLE);
}

void ZavyalovAReduceMPI::MyReduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype type, MPI_Op operation,
                                  int root, MPI_Comm comm) {
  if (operation == MPI_SUM) {
    if (type == MPI_INT) {
      ReduceSumInt(sendbuf, recvbuf, count, root, comm);
    } else if (type == MPI_FLOAT) {
      ReduceSumFloat(sendbuf, recvbuf, count, root, comm);
    } else if (type == MPI_DOUBLE) {
      ReduceSumDouble(sendbuf, recvbuf, count, root, comm);
    }
  } else if (operation == MPI_MIN) {
    if (type == MPI_INT) {
      ReduceMinInt(sendbuf, recvbuf, count, root, comm);
    } else if (type == MPI_FLOAT) {
      ReduceMinFloat(sendbuf, recvbuf, count, root, comm);
    } else if (type == MPI_DOUBLE) {
      ReduceMinDouble(sendbuf, recvbuf, count, root, comm);
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
