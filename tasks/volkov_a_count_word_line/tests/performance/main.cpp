#include <gtest/gtest.h>
#include <mpi.h>

#include <cctype>
#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector> // Added for std::vector

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "volkov_a_count_word_line/common/include/common.hpp"
#include "volkov_a_count_word_line/mpi/include/ops_mpi.hpp"
#include "volkov_a_count_word_line/seq/include/ops_seq.hpp"

namespace volkov_a_count_word_line {

// Changed InType to std::vector<char>
class VolkovACountWordLinePerfTests : public ppc::util::BaseRunPerfTests<std::vector<char>, OutType> {
 protected:
  void SetUp() override {
    std::string file_path =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_volkov_a_count_word_line, "Tolstoy Leo. War and Peace.txt");

    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open input file: " + file_path);
    }

    auto file_size = file.tellg();
    // Read directly into vector
    std::vector<char> single_file_content(static_cast<size_t>(file_size));
    file.seekg(0);
    file.read(single_file_content.data(), file_size);
    file.close();

    const int repetition_count = 10;
    input_data_.reserve(single_file_content.size() * repetition_count);
    for (int i = 0; i < repetition_count; ++i) {
      input_data_.insert(input_data_.end(), single_file_content.begin(), single_file_content.end());
    }

    // Logic verification on the vector data
    auto is_valid_char = [](char c) {
      return (std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '-' || c == '_';
    };

    size_t idx = 0;
    size_t len = input_data_.size();
    int count = 0;

    while (idx < len) {
      while (idx < len && !is_valid_char(input_data_[idx])) {
        idx++;
      }
      if (idx < len) {
        count++;
        while (idx < len && is_valid_char(input_data_[idx])) {
          idx++;
        }
      }
    }

    expected_output_ = count;
  }

  std::vector<char> GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &val) override {
    int rank = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank == 0) {
      return val == expected_output_;
    }
    return true;
  }

 private:
  std::vector<char> input_data_;
  OutType expected_output_ = 0;
};

TEST_P(VolkovACountWordLinePerfTests, RunPerformance) {
  ExecuteTest(GetParam());
}

// Update MakeAllPerfTasks to use std::vector<char>
const auto kPerfTasks = ppc::util::MakeAllPerfTasks<std::vector<char>, VolkovACountWordLineMPI, VolkovACountWordLineSEQ>(
    PPC_SETTINGS_volkov_a_count_word_line);

const auto kTestParams = ppc::util::TupleToGTestValues(kPerfTasks);

const auto kTestNameGen = VolkovACountWordLinePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(VolkovPerfTests, VolkovACountWordLinePerfTests, kTestParams, kTestNameGen);

}  // namespace volkov_a_count_word_line