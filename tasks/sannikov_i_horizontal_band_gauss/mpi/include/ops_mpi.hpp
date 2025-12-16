#pragma once

#include "sannikov_i_horizontal_band_gauss/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sannikov_i_horizontal_band_gauss {

class SannikovIHorizontalBandGaussMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SannikovIHorizontalBandGaussMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BuildRowPartition(int size, int n, std::vector<int> *counts, std::vector<int> *displs);
};

}  // namespace sannikov_i_horizontal_band_gauss
