#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "zavyalov_a_reduce/common/include/common.hpp"

namespace zavyalov_a_reduce {

namespace {  // внутренние helper-ы

template <typename T>
inline void apply_sum(std::vector<T> &acc, const std::vector<T> &temp, int count) {
  for (int i = 0; i < count; i++) {
    acc[i] += temp[i];
  }
}

template <typename T>
inline void apply_min(std::vector<T> &acc, const std::vector<T> &temp, int count) {
  for (int i = 0; i < count; i++) {
    acc[i] = std::min(acc[i], temp[i]);
  }
}

template <typename T>
void reduce_binary_tree(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm, MPI_Datatype type,
                        void (*apply_op)(std::vector<T> &, const std::vector<T> &, int)) {
  int world_size = 0, world_rank = 0;
  MPI_Comm_size(comm, &world_size);
  MPI_Comm_rank(comm, &world_rank);

  std::vector<T> acc(count);
  std::vector<T> tmp(count);

  std::memcpy(acc.data(), sendbuf, sizeof(T) * count);

  for (int offset = 1; offset < world_size; offset <<= 1) {
    int group_leader = world_rank % (2 * offset);

    if (group_leader == 0) {
      int src = world_rank + offset;
      if (src < world_size) {
        MPI_Recv(tmp.data(), count, type, src, src, comm, MPI_STATUS_IGNORE);
        apply_op(acc, tmp, count);
      }
    } else {
      MPI_Send(acc.data(), count, type, world_rank - offset, world_rank, comm);
      return;  // участник отправил данные и вышел
    }
  }

  // Перемещение результата на root
  if (world_rank == 0) {
    if (root == 0) {
      std::memcpy(recvbuf, acc.data(), sizeof(T) * count);
    } else {
      MPI_Send(acc.data(), count, type, root, 0, comm);
    }
  } else if (world_rank == root) {
    MPI_Recv(recvbuf, count, type, 0, 0, comm, MPI_STATUS_IGNORE);
  }
}

template <typename T>
void ReduceSumImpl(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm, MPI_Datatype type) {
  reduce_binary_tree<T>(sendbuf, recvbuf, count, root, comm, type, apply_sum<T>);
}

template <typename T>
void ReduceMinImpl(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm, MPI_Datatype type) {
  reduce_binary_tree<T>(sendbuf, recvbuf, count, root, comm, type, apply_min<T>);
}

}  // namespace

// ===== API обёртки =====

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

// ===== универсальная обёртка =====

void ZavyalovAReduceMPI::MyReduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype type, MPI_Op operation,
                                  int root, MPI_Comm comm) {
  if (operation == MPI_SUM) {
    if (type == MPI_INT) {
      ReduceSumInt(sendbuf, recvbuf, count, root, comm);
    } else if (type == MPI_FLOAT) {
      ReduceSumFloat(sendbuf, recvbuf, count, root, comm);
    } else {
      ReduceSumDouble(sendbuf, recvbuf, count, root, comm);
    }
  } else if (operation == MPI_MIN) {
    if (type == MPI_INT) {
      ReduceMinInt(sendbuf, recvbuf, count, root, comm);
    } else if (type == MPI_FLOAT) {
      ReduceMinFloat(sendbuf, recvbuf, count, root, comm);
    } else {
      ReduceMinDouble(sendbuf, recvbuf, count, root, comm);
    }
  }
}

// ===== Встроенные методы фреймворка =====

ZavyalovAReduceMPI::ZavyalovAReduceMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::get<0>(GetOutput()) = std::shared_ptr<void>(nullptr);
}

bool ZavyalovAReduceMPI::ValidationImpl() {
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (rank != 0) {
    return true;
  }

  bool ok = true;
  MPI_Op op = std::get<0>(GetInput());
  ok &= (op == MPI_SUM || op == MPI_MIN);

  MPI_Datatype type = std::get<1>(GetInput());
  ok &= (type == MPI_INT || type == MPI_FLOAT || type == MPI_DOUBLE);

  size_t sz = std::get<2>(GetInput());
  ok &= (sz > 0);

  auto ptr = std::get<3>(GetInput());
  ok &= (ptr != nullptr);

  int root = std::get<4>(GetInput());
  ok &= (root < world_size);

  return ok;
}

bool ZavyalovAReduceMPI::PreProcessingImpl() {
  return true;
}

bool ZavyalovAReduceMPI::RunImpl() {
  MPI_Op op = std::get<0>(GetInput());
  MPI_Datatype type = std::get<1>(GetInput());
  size_t sz = std::get<2>(GetInput());
  auto mem_ptr = std::get<3>(GetInput());
  void *mem = mem_ptr.get();
  int root = std::get<4>(GetInput());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int type_size = 0;
  MPI_Type_size(type, &type_size);

  auto raw_output = new char[sz * type_size];
  std::shared_ptr<void> out_ptr(raw_output, [](void *p) { delete[] static_cast<char *>(p); });

  if (rank == root) {
    MyReduce(mem, raw_output, static_cast<int>(sz), type, op, root, MPI_COMM_WORLD);
    MPI_Bcast(raw_output, static_cast<int>(sz), type, root, MPI_COMM_WORLD);
  } else {
    MyReduce(mem, nullptr, static_cast<int>(sz), type, op, root, MPI_COMM_WORLD);
    MPI_Bcast(raw_output, static_cast<int>(sz), type, root, MPI_COMM_WORLD);
  }

  std::get<0>(GetOutput()) = out_ptr;
  std::get<1>(GetOutput()) = false;  // MPI version

  return true;
}

bool ZavyalovAReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_reduce
