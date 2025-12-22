#pragma once

#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_gausse_vert_strip {

class ShvetsovaKGaussVertStripMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShvetsovaKGaussVertStripMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  int size_of_rib_ = 0;
  // Доп функции
  static int GetOwnerOfColumn(int k, int N, int size);
  static int GetColumnStartIndex(int rank, int N, int size);
  void ForwardStep(int k, int N, int local_cols, int col_start, std::vector<std::vector<double>> &A_local,
                   std::vector<double> &b);
  void BackwardStep(int k, int N, int col_start, std::vector<std::vector<double>> &A_local, std::vector<double> &b,
                    std::vector<double> &x);
};

}  // namespace shvetsova_k_gausse_vert_strip
