#pragma once

#include <string>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"

namespace sizov_d_string_mismatch_count {

class StringMismatchCountSequential : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit StringMismatchCountSequential(const InType& input);
  ~StringMismatchCountSequential() override = default;

  StringMismatchCountSequential(const StringMismatchCountSequential&) = delete;
  StringMismatchCountSequential& operator=(const StringMismatchCountSequential&) = delete;

 protected:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string str_a_;
  std::string str_b_;
  int result_ = 0;
};

}  // namespace sizov_d_string_mismatch_count
