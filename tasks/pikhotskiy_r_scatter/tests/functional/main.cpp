// NOLINTBEGIN
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <cstdint>
// <memory> удален, так как не используется

#include "pikhotskiy_r_scatter/common/include/common.hpp"
#include "pikhotskiy_r_scatter/mpi/include/ops_mpi.hpp"
#include "pikhotskiy_r_scatter/seq/include/ops_seq.hpp"

namespace pikhotskiy_r_scatter {

class PikhotskiyRScatterFunctionalTest : public ::testing::Test {
 protected:
  InputType test_input_{};             // NOLINT
  std::vector<int> send_buffer_{};     // NOLINT
  std::vector<int> receive_buffer_{};  // NOLINT

  // NOLINTNEXTLINE
  void SetUp() override {
    const int elements_per_process = 50;

    receive_buffer_.resize(elements_per_process);
    send_buffer_.resize(elements_per_process);

    for (int idx = 0; idx < elements_per_process; ++idx) {
      send_buffer_[idx] = idx * 2 + 5;  // NOLINT
    }

    test_input_.source_buffer = send_buffer_.data();
    test_input_.elements_to_send = elements_per_process;
    test_input_.send_data_type = MPI_INT;
    test_input_.destination_buffer = receive_buffer_.data();
    test_input_.elements_to_receive = elements_per_process;
    test_input_.receive_data_type = MPI_INT;
    test_input_.root_process = 0;
    test_input_.communicator = MPI_COMM_WORLD;
  }
};

// NOLINTNEXTLINE
TEST_F(PikhotskiyRScatterFunctionalTest, SeqVersionWorks) {
  PikhotskiyRScatterSEQ task(test_input_);  // NOLINT

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();
  EXPECT_FALSE(output.empty());

  // ИСПРАВЛЕНО: приведение типов для устранения предупреждения
  EXPECT_EQ(output.size(), static_cast<size_t>(test_input_.elements_to_send) * sizeof(int));

  const int *result_data = reinterpret_cast<const int *>(output.data());
  for (int i = 0; i < test_input_.elements_to_send; ++i) {
    // NOLINTNEXTLINE
    EXPECT_EQ(result_data[i], i * 2 + 5);
  }
}

#ifdef MPI_VERSION
// NOLINTNEXTLINE
TEST_F(PikhotskiyRScatterFunctionalTest, MpiVersionWorks) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
  }

  PikhotskiyRScatterMPI task(test_input_);  // NOLINT

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();
  EXPECT_FALSE(output.empty());
}
#endif

// NOLINTNEXTLINE
TEST(PikhotskiyRScatterTest, BasicStructure) {
  ScatterArguments args;
  args.elements_to_send = 10;
  args.elements_to_receive = 10;

  EXPECT_EQ(args.elements_to_send, 10);
  EXPECT_EQ(args.elements_to_receive, 10);
}

// NOLINTNEXTLINE
TEST(PikhotskiyRScatterTest, VectorType) {
  OutputType vec(100);          // NOLINT
  EXPECT_EQ(vec.size(), 100U);  // ИСПРАВЛЕНО: добавлен суффикс U для беззнакового литерала

  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<std::uint8_t>(i);
  }

  for (size_t i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], static_cast<std::uint8_t>(i));
  }
}

}  // namespace pikhotskiy_r_scatter
// NOLINTEND
