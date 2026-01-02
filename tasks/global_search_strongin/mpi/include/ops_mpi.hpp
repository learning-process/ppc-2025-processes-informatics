#pragma once

#include <utility>
#include <vector>

#include "global_search_strongin/common/include/common.hpp"
#include "task/include/task.hpp"

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

  double ComputeGlobalSlope() const;
  std::pair<int, int> IntervalRange(int intervals) const;
  std::pair<double, int> EvaluateIntervals(int start, int end, double m) const;
  bool TryInsertPoint(const InType &input, int best_index, double epsilon, double m, int &insert_index,
                      double &new_point, double &new_value);
  void BroadcastInsertionData(int &continue_flag, int &insert_index, double &new_point, double &new_value);
  bool ProcessIteration(const InType &input, double reliability, double epsilon);

  int rank_ = 0;
  int world_size_ = 1;
  std::vector<SamplePoint> points_;
  double best_x_ = 0.0;
  double best_value_ = 0.0;
  int iterations_done_ = 0;
};

}  // namespace global_search_strongin
