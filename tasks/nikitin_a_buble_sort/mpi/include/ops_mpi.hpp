#pragma once

#include "nikitin_a_buble_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitin_a_buble_sort {

class NikitinABubleSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NikitinABubleSortMPI(const InType &in);

 private:
  std::vector<double> data;
  int n;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void LocalSort(std::vector<double> &local, int global_offset, int phase_parity);
  void ExchangeRight(std::vector<double> &local, const std::vector<int> &counts, const std::vector<int> &displs,
                     int rank, int comm_size, int phase_parity, int tag);
  void ExchangeLeft(std::vector<double> &local, const std::vector<int> &counts, const std::vector<int> &displs,
                    int rank, int phase_parity, int tag);
};

}  // namespace nikitin_a_buble_sort
