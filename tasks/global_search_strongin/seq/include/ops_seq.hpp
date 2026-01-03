#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "global_search_strongin/common/include/common.hpp"
#include "task/include/task.hpp"

namespace global_search_strongin {

class StronginSearchSeq : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit StronginSearchSeq(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] double ComputeMaxSlope() const;
  [[nodiscard]] std::optional<std::size_t> SelectInterval(double m) const;
  bool InsertPoint(const InType &input, std::size_t interval_index, double epsilon, double m, double left_bound,
                   double right_bound);

  std::vector<SamplePoint> points_;
  double best_x_ = 0.0;
  double best_value_ = 0.0;
  int iterations_done_ = 0;
};

}  // namespace global_search_strongin
