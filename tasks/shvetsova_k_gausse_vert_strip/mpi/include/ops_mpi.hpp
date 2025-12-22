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
  static int GetOwnerOfColumn(int k, int n, int size);
  static int GetColumnStartIndex(int rank, int n, int size);
  void ForwardStep(int k, int n, int local_cols, int col_start, std::vector<std::vector<double>> &a_local,
                   std::vector<double> &b) const;
  void BackwardStep(int k, int n, int col_start, std::vector<std::vector<double>> &a_local, std::vector<double> &b,
                    std::vector<double> &x) const;
};

}  // namespace shvetsova_k_gausse_vert_strip
