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

  // Каждый процесс держит текущий накопленный буфер (если понадобится отправлять дальше)
  std::vector<T> acc_buf(count);
  std::memcpy(acc_buf.data(), sendbuf, static_cast<std::size_t>(count) * sizeof(T));

  // Постепенное слияние: offset = 1,2,4,...
  for (int offset = 1; offset < world_size; offset *= 2) {
    if ((world_rank % (2 * offset)) == 0) {
      int src = world_rank + offset;
      if (src < world_size) {
        MPI_Recv(temp_buf.data(), count, type, src, src, comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; ++i) {
          acc_buf[i] += temp_buf[i];
        }
      }
    } else {
      int dst = world_rank - offset;
      MPI_Send(acc_buf.data(), count, type, dst, world_rank, comm);
      // После отправки процесс выходит из цикла — он больше не участвует
      break;
    }
  }

  // В результате сборка по умолчанию на rank 0
  if (world_rank == 0) {
    if (root != 0) {
      MPI_Send(acc_buf.data(), count, type, root, 0, comm);
    } else {
      std::memcpy(recvbuf, acc_buf.data(), static_cast<std::size_t>(count) * sizeof(T));
    }
  } else if (world_rank == root && root != 0) {
    // root должен получить от 0 итог, если root != 0
    MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
  }
}

template <typename T>
void ReduceMinImpl(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm, MPI_Datatype type) {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(comm, &world_size);
  MPI_Comm_rank(comm, &world_rank);

  std::vector<T> temp_buf(count);

  std::vector<T> acc_buf(count);
  std::memcpy(acc_buf.data(), sendbuf, static_cast<std::size_t>(count) * sizeof(T));

  for (int offset = 1; offset < world_size; offset *= 2) {
    if ((world_rank % (2 * offset)) == 0) {
      int src = world_rank + offset;
      if (src < world_size) {
        MPI_Recv(temp_buf.data(), count, type, src, src, comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; ++i) {
          acc_buf[i] = std::min(acc_buf[i], temp_buf[i]);
        }
      }
    } else {
      int dst = world_rank - offset;
      MPI_Send(acc_buf.data(), count, type, dst, world_rank, comm);
      break;
    }
  }

  if (world_rank == 0) {
    if (root != 0) {
      MPI_Send(acc_buf.data(), count, type, root, 0, comm);
    } else {
      std::memcpy(recvbuf, acc_buf.data(), static_cast<std::size_t>(count) * sizeof(T));
    }
  } else if (world_rank == root && root != 0) {
    MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
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
