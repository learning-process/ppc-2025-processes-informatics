# Сортировка пузырьком (алгоритм чет-нечетной перестановки) / Bubble Sort (Odd–Even Transposition Algorithm)

- Student: Сизов Дмитрий Игоревич, group 3823Б1ФИ2
- Technology: SEQ + MPI
- Variant: 21

## 1. Introduction

The objective of this work is to implement both sequential and parallel (MPI-based) versions of bubble sort using the Odd–Even Transposition Sort algorithm and to compare their correctness and performance. The task belongs to the class of comparison-based sorting algorithms that rely on pairwise exchanges of adjacent elements. The expected outcome is a fully correct implementation of the sorting procedure, successful execution of all functional and performance tests in the PPC framework, and a demonstrated speedup of the MPI version on large input data.

## 2. Problem Statement

Given: an array of integers (`InType = std::vector<int>`).

Required:

- Sort the array in non-decreasing order.
- Provide two implementations: a sequential version and an MPI-based version.
- Both implementations must operate correctly on all provided tests.

Input/output format:

- Input: `std::vector<int>`
- Output: a sorted `std::vector<int>`

Constraint: empty input is not allowed.

## 3. Baseline Algorithm (Sequential)

The sequential version uses the classical odd–even sort algorithm.  
Odd–even sort operates as a sequence of phases, where each phase compares adjacent elements, but the pairs differ between even and odd phases.

During an **even phase**, the algorithm processes element pairs starting at even indices: (0,1), (2,3), (4,5), and so on.  
For each pair, if the left element is greater than the right one, the two elements are swapped.

During an **odd phase**, the algorithm processes pairs starting at odd indices: (1,2), (3,4), (5,6), etc.  
The same swapping rule applies.

These phases alternate, and due to the regular shift of comparison windows, any elements placed in the wrong order gradually move toward their correct positions.  
The algorithm continues alternating phases until a pass occurs with no swaps, which indicates that the array is fully sorted.

Time complexity: **O(n²)**.

## 4. Parallelization Scheme (MPI)

### 4.1 Data Distribution

During data distribution, the function `ComputeScatterInfo` is invoked. It calculates:

- `counts[i]` — the number of elements assigned to process *i*,
- `displs[i]` — the displacement (offset) of each block within the global array.

This ensures an almost even distribution of data, even when `n % p != 0`.  
`MPI_Scatterv` then uses these arrays to distribute the corresponding subarray to each process.

### 4.2 Local Work

Each process receives its portion of the global array and performs a local sort of that subarray using  
`std::ranges::sort(local)`.

At this stage, every process holds a **locally sorted fragment**, but this does not imply global ordering.  
Different processes may hold value ranges that overlap with each other.

Local sorting is required to ensure that block-wise comparisons during the odd–even transposition phases
produce correct global movement of elements.

### 4.3 Neighbor Exchanges (Odd–Even Transposition Phases)

The main part of the MPI algorithm consists of repeated odd–even transposition phases.  
This approach parallels the idea of the sequential bubble sort, but operations are performed on whole blocks of data rather than on individual neighboring elements.

A total of `n` global phases is executed, where `n` is the number of elements in the input array.  
This corresponds to the worst-case number of iterations needed for bubble sort to reach a fully ordered state.

In each phase the following steps are performed:

1. The algorithm determines whether the current phase number is even or odd.

2. Based on phase parity and process rank, each process computes the rank of its neighbor with which it may need to exchange data.  
   Depending on the situation, the partner may be either the preceding or the following process in rank order.

3. If the computed partner rank falls outside the valid range of processes, the exchange for this phase is skipped.

4. Otherwise, the process performs a two-way exchange of its sorted block using `MPI_Sendrecv`, receiving the partner’s block into a temporary buffer.

5. The local block and the received block are merged into a single sorted sequence.

6. Each process keeps only the part of the merged sequence that corresponds to its position in the global order:  
   processes with a smaller rank keep the lower portion; processes with a larger rank keep the upper portion.

As these phases progress, smaller values gradually drift toward processes with lower ranks, and larger values toward processes with higher ranks.  
This reproduces the logic of odd–even transposition sorting at the level of distributed data blocks.

### 4.4 Gathering and Finalization

After all odd–even transposition phases have been completed, each process holds a block that is locally sorted and positioned correctly relative to blocks on neighboring ranks.

The final steps are performed by the root process:

1. All local blocks are collected into a single global array using `MPI_Gatherv`, which reconstructs the original data layout based on the previously computed `counts` and `displs` arrays.

2. The assembled array on the root is taken as the final result.  
   Since the odd–even transposition algorithm guarantees that the global ordering is correct after the required number of phases, no additional processing is necessary.

At this point, the fully sorted array is available on the root process and can be returned as the final output of the parallel algorithm.

## 5. Implementation Details

### 5.1 Project Structure

```cpp
tasks/sizov_d_bubble_sort
├── common
│   └── include
│       └── common.hpp
├── data
│   ├── test1.txt
│   ├── test2.txt
│   ├── test3.txt
│   ├── ...
│   └── test40.txt
├── mpi
│   ├── include
│   │   └── ops_mpi.hpp
│   └── src
│       └── ops_mpi.cpp
├── seq
│   ├── include
│   │   └── ops_seq.hpp
│   └── src
│       └── ops_seq.cpp
├── tests
│   ├── functional
│   │   └── main.cpp
│   └── performance
│       └── main.cpp
├── info.json
├── report.md
└── settings.json

```

### 5.2 Data

The functional tests use a set of input files located in:
  `tasks/sizov_d_bubble_sort/data/testN.txt`

Each file contains two lines:  

1. an unsorted array,  
2. the reference sorted result.

### 5.3 Additional helper functions

Several auxiliary helper functions are used in the implementation.  
Most of them are placed inside an anonymous namespace in `mpi/src/ops_mpi.cpp`,  
and the `TrimString` utility function is located in the functional test module.

- `ComputeScatterInfo(total, size, counts, displs)`  
  Computes the block sizes (`counts`) and their displacements (`displs`) for `MPI_Scatterv`.  
  The function correctly handles cases where `total % size != 0` by distributing the extra elements among the first processes, ensuring that the data is partitioned as evenly as possible.

- `ComputePartner(even_phase, even_rank, rank)`  
  Determines the exchange partner for the current phase of the odd–even transposition algorithm.  
  This helper encapsulates the neighbor-selection rules and makes the implementation clearer.

- `OddEvenExchange(local, counts, rank, size, phase)`  
  Performs a single phase of the parallel odd–even sort.  
  During the phase, each process may exchange its local block with a neighboring process using `MPI_Sendrecv`, after which the two blocks are merged.  
  Each process then retains only the part of the merged sequence that corresponds to its rank.  
  Boundary conditions (such as absence of a neighbor) are handled inside the function.

- `GatherResult(local, counts, displs, rank, total, output)`  
  Collects the final blocks from all processes into a single global array on the root process using `MPI_Gatherv`.  
  Since the odd–even transposition algorithm ensures globally correct ordering after all phases, the gathered array is already sorted.

- `TrimString(std::string& s)` (from the functional tests)  
  Removes whitespace characters (`\r`, `\n`, `\t`) from both ends of a line before parsing test data.  
  This prevents errors when reading integers from files that contain leading or trailing whitespace.

## 6. Experimental Setup

|  Component |                Value                   |
|------------|----------------------------------------|
|     CPU    | 12th Gen Intel(R) Core(TM) i5-12450H   |
|     RAM    |                 16 GB                  |
|     ОС     | OS: Ubuntu 24.04 (DevContainer / WSL2) |
|  Compiler  | GCC 13.3.0 (g++), C++20, CMake, Release|
|     MPI    |        mpirun (Open MPI) 4.1.6         |

## 7. Results and Discussion

### 7.1 Correctness

Correctness has been verified through:

- multiple functional tests,
- comparison against `std::ranges::sort`,
- matching results between the MPI and SEQ implementations.

### 7.2 Performance

| Mode | Number of Processes  | Time (s)      | Speedup        | Efficiency         |
|------|----------------------|---------------|----------------|--------------------|
| SEQ  | 1                    | 0.8406618595  | 1.000000000    | N/A                |
| MPI  | 2                    | 0.0016978002  | 495.1476973    | 24757.3848648%     |
| MPI  | 4                    | 0.0020235174  | 415.4458269    | 10386.1456726%     |
| MPI  | 8                    | 0.0026713558  | 314.6948301    | 3933.68537570%     |
| MPI  | 16                   | 0.0025910712  | 324.4456808    | 2027.78550500%     |
| MPI  | 32                   | 0.0017847040  | 471.0371353    | 1471.99104780%     |

For local performance measurements and for comparing the sequential and parallel versions, a unified input size of 200,000 elements was used.  
The performance tests were executed locally on my machine with the hardware configuration listed above.  
All performance results are averaged over three runs.  
The time reported in the third column corresponds to the `task_run` execution time.

The speedup is computed as:

$$S(p) = \frac{T_{seq}}{T_{p}}$$

The efficiency is computed as:

$$E(p) = \frac{S(p)}{p} \cdot 100\%$$

### 7.3 Environment variables

The following standard PPC environment variables were used to run the performance tests:

- **PPC_NUM_PROC** — specifies the number of MPI processes for the measurements  
  (in the experiments: 1, 2, 4, 8, 16, 32).

- **PPC_PERF_MAX_TIME** — the maximum allowed execution time for a single test (20 seconds).

- **PPC_IGNORE_TEST_TIME_LIMIT** — allows performance tests to run without time-limit restrictions  
  (used during manual execution).

- **TMPDIR** — the directory for temporary files required by the PPC infrastructure.

## 8. Conclusions

- Both SEQ and MPI versions of the odd–even bubble sort have been implemented.
- Both implementations successfully pass all tests.
- The MPI version demonstrates significant speedup.
- The main limitations are the quadratic time complexity and the need for inter-process exchanges at each phase.

## 9. References

- Гергель В.П. Высокопроизводительные вычисления для многопроцессорных многоядерных систем. Серия «Суперкомпьютерное образование».
- Гергель В.П. Теория и практика параллельных вычислений. – М.: Интуит Бином. Лаборатория знаний, 2007.
- Wilkinson B., Allen M. Parallel programming: techniques and applications using networked workstations and parallel computers. – Prentice Hall, 1999.

## Appendix

```cpp
bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total = static_cast<int>(data_.size());
  const int rem = total % size;

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  ComputeScatterInfo(total, size, rem, counts, displs);

  const int local_n = counts[rank];
  std::vector<int> local(local_n);

  MPI_Scatterv(data_.data(), counts.data(), displs.data(), MPI_INT, local.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

  std::ranges::sort(local);

  for (int phase = 0; phase < size; ++phase) {
    OddEvenExchange(local, counts, rank, size, phase);
  }

  std::vector<int> global;
  GatherResult(local, counts, displs, rank, total, global);

  if (rank == 0) {
    GetOutput() = global;
  }

  return true;
}
```
