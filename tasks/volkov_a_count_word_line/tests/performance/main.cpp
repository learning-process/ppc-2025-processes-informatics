#include <gtest/gtest.h>
#include <mpi.h>

#include <cctype>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "volkov_a_count_word_line/common/include/common.hpp"
#include "volkov_a_count_word_line/mpi/include/ops_mpi.hpp"
#include "volkov_a_count_word_line/seq/include/ops_seq.hpp"

namespace volkov_a_count_word_line {

class VolkovACountWordLinePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    std::string file_path =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_volkov_a_count_word_line, "Tolstoy Leo. War and Peace.txt");

    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open input file: " + file_path);
    }

    auto file_size = file.tellg();
    input_data_.resize(static_cast<size_t>(file_size));
    file.seekg(0);
    file.read(&input_data_[0], file_size);
    file.close();

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

  InType GetTestInputData() override {
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
  InType input_data_;
  OutType expected_output_ = 0;
};

TEST_P(VolkovACountWordLinePerfTests, RunPerformance) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VolkovACountWordLineMPI, VolkovACountWordLineSEQ>(PPC_SETTINGS_volkov_a_count_word_line);

const auto kTestParams = ppc::util::TupleToGTestValues(kPerfTasks);

const auto kTestNameGen = VolkovACountWordLinePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(VolkovPerfTests, VolkovACountWordLinePerfTests, kTestParams, kTestNameGen);

}  // namespace volkov_a_count_word_line