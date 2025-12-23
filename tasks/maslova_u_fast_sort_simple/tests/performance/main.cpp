#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "maslova_u_fast_sort_simple/common/include/common.hpp"
#include "maslova_u_fast_sort_simple/mpi/include/ops_mpi.hpp"
#include "maslova_u_fast_sort_simple/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace maslova_u_fast_sort_simple {

class MaslovaUFastSortPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int rank = 0;
    int is_mpi = 0;
    MPI_Initialized(&is_mpi);
    if (is_mpi) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank == 0) {
      const size_t count = 1000000;
      input_data_.resize(count);
      std::mt19937 gen(42);
      for (size_t i = 0; i < count; ++i) {
        input_data_[i] = static_cast<int>(gen() % 100000);
      }
      expected_output_ = input_data_;
      std::sort(expected_output_.begin(), expected_output_.end());
    } else {
      input_data_.clear();
      expected_output_.clear();
    }
  }
  InType GetTestInputData() final {
    return input_data_;
  }
  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int is_initialized = 0;
    MPI_Initialized(&is_initialized);

    if (is_initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank != 0) {
      return true;
    }
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(static_cast<double>(output_data[i]) - static_cast<double>(expected_output_[i])) > 1e-6) {
        return false;
      }
    }
    return true;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(MaslovaUFastSortPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MaslovaUFastSortSimpleMPI, MaslovaUFastSortSimpleSEQ>(
    PPC_SETTINGS_maslova_u_fast_sort_simple);

INSTANTIATE_TEST_SUITE_P(fastSortPerf, MaslovaUFastSortPerfTests, ppc::util::TupleToGTestValues(kAllPerfTasks),
                         MaslovaUFastSortPerfTests::CustomPerfTestName);

}  // namespace maslova_u_fast_sort_simple
