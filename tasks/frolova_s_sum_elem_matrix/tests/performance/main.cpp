#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"
#include "frolova_s_sum_elem_matrix/seq/include/ops_seq.hpp"

namespace frolova_s_sum_elem_matrix {

// Объявление функции
static std::vector<std::vector<int>> CreateMatrix(int rows, int cols, int value = 1);

// Вспомогательная функция для создания матрицы
std::vector<std::vector<int>> CreateMatrix(int rows, int cols, int value) {
  std::vector<std::vector<int>> matrix(rows);
  for (auto &row : matrix) {
    row.resize(cols, value);
  }
  return matrix;
}

class FrolovaSSumElemMatrixPerformanceTests : public ::testing::Test {
 protected:
  void SetUp() override {
    matrix_1000x1000 = CreateMatrix(1000, 1000);
    matrix_2000x2000 = CreateMatrix(2000, 2000);
  }

  std::vector<std::vector<int>> matrix_1000x1000;
  std::vector<std::vector<int>> matrix_2000x2000;
};

// Тест производительности SEQ версии
TEST_F(FrolovaSSumElemMatrixPerformanceTests, SEQ1000x1000) {
  FrolovaSSumElemMatrixSEQ task(matrix_1000x1000);

  auto start = std::chrono::high_resolution_clock::now();
  bool result = task.Run();
  auto end = std::chrono::high_resolution_clock::now();

  ASSERT_TRUE(result);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "SEQ 1000×1000: " << duration.count() << " ms" << '\n';
}

TEST_F(FrolovaSSumElemMatrixPerformanceTests, SEQ2000x2000) {
  FrolovaSSumElemMatrixSEQ task(matrix_2000x2000);

  auto start = std::chrono::high_resolution_clock::now();
  bool result = task.Run();
  auto end = std::chrono::high_resolution_clock::now();

  ASSERT_TRUE(result);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "SEQ 2000×2000: " << duration.count() << " ms" << '\n';
}

TEST_F(FrolovaSSumElemMatrixPerformanceTests, MPI1000x1000) {
  FrolovaSSumElemMatrixMPI task(matrix_1000x1000);

  auto start = std::chrono::high_resolution_clock::now();
  bool result = task.Run();
  auto end = std::chrono::high_resolution_clock::now();

  ASSERT_TRUE(result);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  int rank = 0;
  int size = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "MPI 1000×1000 (processes: " << size << "): " << duration.count() << " ms" << '\n';
  }
}

TEST_F(FrolovaSSumElemMatrixPerformanceTests, MPI2000x2000) {
  FrolovaSSumElemMatrixMPI task(matrix_2000x2000);

  auto start = std::chrono::high_resolution_clock::now();
  bool result = task.Run();
  auto end = std::chrono::high_resolution_clock::now();

  ASSERT_TRUE(result);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "MPI 2000×2000 (processes: " << size << "): " << duration.count() << " ms" << '\n';
  }
}

}  // namespace frolova_s_sum_elem_matrix
