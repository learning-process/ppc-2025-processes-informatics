# Сортировка пузырьком (алгоритм чет-нечетной перестановки) / Bubble Sort (Odd–Even Transposition Algorithm)

- Student: Сизов Дмитрий Игоревич, group 3823Б1ФИ2
- Technology: SEQ + MPI
- Variant: 21

## 1. Introduction
The objective of this work is to implement both sequential and parallel (MPI-based) versions of bubble sort using the Odd–Even Transposition Sort algorithm and to compare their correctness and performance.
The task belongs to the class of comparison-based sorting algorithms that rely on pairwise exchanges of adjacent elements.

The expected outcome is a fully correct implementation of the sorting procedure, successful execution of all functional and performance tests in the PPC framework, and a demonstrated speedup of the MPI version on large input data.

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
- `displs[i]` — the displacement of each block within the global array.

This ensures an almost even distribution of data, even when `n % p != 0`.  
`MPI_Scatterv` then uses these arrays to send the corresponding subarray to each process.

### 4.2 Local Work
Each process locally sorts its assigned subarray using `std::ranges::sort(local)`.  
At this stage, every process holds a correctly sorted fragment, but global ordering is not yet guaranteed, because different processes may still contain overlapping value ranges.

### 4.3 Neighbor Exchanges
The main part of the algorithm consists of exchanges between neighboring processes.  
The code performs `size` phases — one phase per process.

In each phase:
1. The phase parity is determined: `even_phase = (phase % 2 == 0)`.
2. Each process computes its exchange partner:
   - if the phase is even and the rank is even → partner = rank + 1,
   - if the phase is even and the rank is odd  → partner = rank - 1,
   - if the phase is odd  and the rank is odd  → partner = rank + 1,
   - if the phase is odd  and the rank is even → partner = rank - 1.
3. If the partner index is invalid (`partner < 0` or `partner >= size`), the exchange is skipped.
4. Otherwise, processes exchange their sorted blocks using `MPI_Sendrecv`.  
   Each process receives the partner’s sorted block into `recvbuf`.
5. Both processes merge `local` and `recvbuf` into a single sorted array `merged`.
6. Each process then keeps the appropriate half of the merged result:
   - if `rank < partner` → keep the first `local_n` elements,
   - if `rank > partner` → keep the last `local_n` elements.

This exchange procedure ensures that elements gradually move toward their correct global positions — similar to how odd–even sort pushes larger values to the right and smaller values to the left.

### 4.4 Gathering & Broadcast
After all phases are completed, the root process performs:
1. Gathering of all local blocks into a global array using `MPI_Gatherv`,
2. A final `std::ranges::sort` on the root to guarantee full global ordering,
3. Broadcasting the final sorted array to all processes using `MPI_Bcast`.


## 5. Implementation Details

### 5.1 Project Structure
```
tasks/sizov_d_bubble_sort
├── common
│   └── include
│       └── common.hpp
├── data
│   ├── test1.txt
│   ├── test2.txt
│   ├── test3.txt
│   ├── ...
│   └── testN.txt
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

- `ComputeScatterInfo(total, size, rem, counts, displs)`  
  Computes the block sizes (`counts`) and their displacements (`displs`) for `MPI_Scatterv`.  
  The function correctly handles cases where `total % size != 0` by distributing the “extra” elements among the first processes, ensuring the most balanced data partitioning possible.

- `ComputePartner(even_phase, even_rank, rank)`  
  Determines the exchange partner for the current phase of the odd–even sorting algorithm.  
  The logic fully matches the algorithm: in even phases, even‐rank processes exchange with the right neighbor, and in odd phases, odd‐rank processes do so.  
  This function simplifies partner selection and improves code readability.

- `OddEvenExchange(local, counts, rank, size, phase)`  
  Implements a single phase of the parallel odd–even sort.  
  It performs a send/receive exchange via `MPI_Sendrecv`, merges two sorted blocks, and selects the appropriate half of the merged result.  
  The function also handles boundary cases (e.g., when a neighbor does not exist).

- `GatherResult(local, counts, displs, rank, total, output)`  
  Collects the final results on the root process using `MPI_Gatherv`.  
  After gathering, the root performs a final sort to ensure robustness of the global ordering, especially for boundary regions between distributed blocks.

- `TrimString(std::string& s)` (from the functional tests)  
  Removes whitespace characters (`\r`, `\n`, `\t`) from both ends of a line before parsing test data.  
  This prevents errors when reading integers from input files that contain trailing newline characters.


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

For CI execution, the performance tests use automatically generated input data, with different array sizes for the SEQ and MPI modes:
- for SEQ: an array of 150,000 elements filled in descending order;
- for MPI: an array of 1,000,000 elements, also filled in descending order.

For local performance measurements and for comparing the sequential and parallel versions, a unified input size of 150,000 elements was used.  
The performance tests were executed locally on my machine with the hardware configuration listed above.  
All performance results are averaged over three runs.  
The time reported in the third column corresponds to the `task_run` execution time.

The speedup is computed as:

\[
S(p) = \frac{T_{\text{seq}}}{T_p}
\]

The efficiency is computed as:

\[
E(p) = \frac{S(p)}{p} \times 100\%
\]

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
