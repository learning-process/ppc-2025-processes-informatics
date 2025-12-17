#pragma once

#include <vector>

#include "global_search_strongin/common/include/common.hpp"

namespace global_search_strongin {

class StronginSearchMpi : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit StronginSearchMpi(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rank_ = 0;
  int world_size_ = 1;
  std::vector<SamplePoint> points_;
  double best_x_ = 0.0;
  double best_value_ = 0.0;
  int iterations_done_ = 0;
};

}  // namespace global_search_strongin
