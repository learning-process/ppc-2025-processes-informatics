#pragma once

#include <string>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"

namespace sizov_d_string_mismatch_count {

class StringMismatchCountParallelMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit StringMismatchCountParallelMPI(const InType& input);
  ~StringMismatchCountParallelMPI() override = default;

  StringMismatchCountParallelMPI(const StringMismatchCountParallelMPI&) = delete;
  StringMismatchCountParallelMPI& operator=(const StringMismatchCountParallelMPI&) = delete;

 protected:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string str_a_;
  std::string str_b_;
  int global_result_ = 0;
};

}  // namespace sizov_d_string_mismatch_count
