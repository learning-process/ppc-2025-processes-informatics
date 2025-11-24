#pragma once

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_max_diff_neig_vec {

class ShvetsovaKMaxDiffNeigVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShvetsovaKMaxDiffNeigVecMPI(const InType &in);

 private:
  std::vector<double> data_;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shvetsova_k_max_diff_neig_vec
