#pragma once

#include <mpi.h>

#include "task/include/task.hpp"
#include "zavyalov_a_reduce/common/include/common.hpp"

namespace zavyalov_a_reduce {

class ZavyalovAReduceMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZavyalovAReduceMPI(const InType &in);

 private:
  void ReduceSumInt(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm);
  void ReduceSumFloat(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm);
  void ReduceSumDouble(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm);

  void ReduceMinInt(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm);
  void ReduceMinFloat(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm);
  void ReduceMinDouble(const void *sendbuf, void *recvbuf, int count, int root, MPI_Comm comm);

  void ReduceSum(const void *sendbuf, void *recvbuf, int count, MPI_Datatype type, MPI_Op operation, int root,
                 MPI_Comm comm);
  void MyReduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype type, MPI_Op operation, int root,
                MPI_Comm comm);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zavyalov_a_reduce
