#include "sizov_d_string_mismatch_count/seq/include/ops_seq.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <thread>  // для std::this_thread::get_id()

#include "sizov_d_string_mismatch_count/common/include/common.hpp"

namespace sizov_d_string_mismatch_count {

SizovDStringMismatchCountSEQ::SizovDStringMismatchCountSEQ(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0;
}

bool SizovDStringMismatchCountSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);
  return !a.empty() && a.size() == b.size();
}

bool SizovDStringMismatchCountSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  str_a_ = std::get<0>(input);
  str_b_ = std::get<1>(input);
  result_ = 0;
  return true;
}

bool SizovDStringMismatchCountSEQ::RunImpl() {
  using namespace std::chrono;

  auto start_time = high_resolution_clock::now();
  auto thread_id = std::this_thread::get_id();

  std::cerr << "[SEQ][thread " << thread_id << "] Start RunImpl: len=" << str_a_.size() << "\n";

  result_ = 0;
  for (std::size_t i = 0; i < str_a_.size(); ++i) {
    if (str_a_[i] != str_b_[i]) {
      ++result_;
    }
  }

  auto end_time = high_resolution_clock::now();
  double elapsed_ms = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

  std::cerr << "[SEQ][thread " << thread_id << "] End RunImpl → result=" << result_ << " | time=" << elapsed_ms
            << " ms\n";

  return true;
}

bool SizovDStringMismatchCountSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace sizov_d_string_mismatch_count
