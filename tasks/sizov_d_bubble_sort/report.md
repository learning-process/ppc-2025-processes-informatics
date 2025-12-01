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

At the start of the parallel algorithm, the root process (rank 0) broadcasts  
the total number of elements `n` using `MPI_Bcast`, ensuring all processes  
know the global problem size.

To divide the array among processes, the helper function `ComputeScatterInfo`  
computes two arrays:

- `counts[i]` — number of elements assigned to process *i*  
- `displs[i]` — starting offset of process *i* in the global array

The distribution is nearly uniform: each process receives  
`⌊n / p⌋` or `⌊n / p⌋ + 1` elements, with the first `n % p` processes receiving one extra.

These arrays are passed to `MPI_Scatterv`, which sends each process its  
contiguous block of the input array.

### 4.2 Local Work

After distribution, each process receives its portion of the global array in the vector `local`.

Since the parallel algorithm is based on exchanging already sorted blocks, each process performs a local sort:

    if (local_n > 1) {
        std::ranges::sort(local);
    }

This ensures that the block it owns is internally ordered before participating in the global odd–even transposition.

During subsequent phases, each process:

• keeps its block sorted at all times,  
• exchanges boundary elements with its left or right neighbor when required by the phase parity,  
• merges two sorted blocks and retains only the portion corresponding to its rank (lower half for left process, upper half for right process).

Local sorting happens only once; the resulting global order emerges progressively across odd–even phases.

### 4.3 Neighbor Exchanges (Odd–Even Transposition Phases)

The MPI algorithm implements a distributed variant of **odd–even transposition sort**, analogous to the sequential bubble sort but operating on blocks of data rather than individual elements.

A total of **`n` global phases** is executed, where `n` is the size of the input array.
This matches the upper bound on bubble-sort–style passes required to guarantee complete sorting.

Each global phase performs the following actions:

1. **Determine phase parity**

   The algorithm checks whether the current phase number is even or odd.
   This defines which global index pairs must be compared during the phase.

2. **Compute the neighbor process (partner)**

   Based on:
   • phase parity,  
   • process rank,  
   • block boundaries (`displs`),

   each process determines whether it should communicate with the previous rank or the next rank.

   If the computed partner lies outside the valid range `[0, p-1]`, the process skips this phase.

3. **Exchange boundary elements with the partner**

   When a valid partner exists, the two processes exchange the boundary element(s) using `MPI_Sendrecv`.
   Each process sends:
   • either its local maximum (right boundary),  
   • or its local minimum (left boundary),  
   depending on the direction of exchange.

4. **Resolve boundary conflicts**

   After receiving a value from the partner process, each process updates its own boundary:
   • processes on the “left” keep the minimum,  
   • processes on the “right” keep the maximum.

   This mimics the logic of bubble sort: small values drift left, large values drift right.

As phases progress, values gradually move across process boundaries toward their globally correct positions.
After enough phases, all elements across all processes become fully sorted.

### 4.4 Gathering and Finalization

After all odd–even transposition phases are completed, each process holds a block that is both **locally sorted** and **globally positioned correctly** relative to neighboring processes.

The finalization consists of two steps handled by the root process:

1. **Collecting results**

   All processes send their local sorted blocks to the root using  
   `MPI_Gatherv`, which reconstructs the full array according to the
   previously computed `counts` and `displs`.  
   This ensures that each block is placed back into its original position
   in the global sequence.

2. **Producing the final output**

   The array assembled on the root process represents the final sorted
   sequence.  
   Since odd–even transposition guarantees global correctness after the full
   set of phases, no further adjustments are required.

At this stage, the fully ordered array is available on the root process and becomes the final result of the parallel sorting procedure.

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

### 5.3 Additional Helper Functions

Several auxiliary helper functions are used in the implementation.  
Most of them reside inside an anonymous namespace in `mpi/src/ops_mpi.cpp`,  
while the `TrimString` utility function is located in the functional test module.

- **`ComputeScatterInfo(total, size, counts, displs)`**  
  Computes the number of elements (`counts[i]`) assigned to each process *i* and  
  the starting offset (`displs[i]`) of each block in the global array.  
  The function correctly handles the case `total % size != 0` by giving one extra element  
  to the first few processes, ensuring the most balanced distribution possible.

- **`ComputePartner(even_phase, even_rank, rank)`**  
  Determines the neighbor process with which the current process should exchange data in this phase.  
  This helper encapsulates the partner-selection logic, improving clarity inside the main loop.

- **`OddEvenExchange(local, counts, rank, size, phase)`**  
  Executes a single odd–even transposition phase.  
  Depending on rank and phase parity, a process may exchange boundary elements with its left or right neighbor using `MPI_Sendrecv`.  
  After receiving the partner’s element, the process updates either the left or right boundary accordingly.  
  Boundary checks and invalid partner cases are handled internally.

- **`GatherResult(local, counts, displs, rank, total, output)`**  
  Collects all local blocks into a single global array on the root process using `MPI_Gatherv`.  
  Because odd–even transposition ensures global correctness after all phases, the final gathered array is already sorted.

- **`TrimString(std::string& s)`** *(from functional test utilities)*  
  Removes leading and trailing whitespace characters such as `\r`, `\n`, and `\t` before parsing input lines from test files.  
  This prevents parsing issues caused by stray whitespace in test data.

## 6. Experimental Setup

| Component  | Value                                   |
|------------|-----------------------------------------|
| CPU        | 12th Gen Intel(R) Core(TM) i5-12450H    |
| RAM        | 16 GB                                   |
| ОС         | OS: Ubuntu 24.04 (DevContainer / WSL2)  |
| Compiler   | GCC 13.3.0 (g++), C++20, CMake, Release |
| MPI        | mpirun (Open MPI) 4.1.6                 |

## 7. Results and Discussion

### 7.1 Correctness

Correctness has been verified through:

- multiple functional tests,
- comparison against `std::ranges::sort`,
- matching results between the MPI and SEQ implementations.

### 7.2 Performance

Performance (pipeline):

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 1.8453556061   | 1.0000       | N/A             |
| MPI  | 2         | 2.7543675592   | 0.6700       | 33.50%          |
| MPI  | 4         | 1.9132137878   | 0.9645       | 24.11%          |
| MPI  | 8         | 2.9345040332   | 0.6287       | 7.86%           |

Performance (task_run):

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 1.8453556061   | 1.0000       | N/A             |
| MPI  | 2         | 2.6698493930   | 0.6913       | 34.56%          |
| MPI  | 4         | 1.9826194318   | 0.9307       | 23.26%          |
| MPI  | 8         | 3.1027069148   | 0.5946       | 7.43%           |

For local performance measurements and for comparing the sequential and parallel versions, a unified input size of 100,000 elements was used.  
The performance tests were executed locally on my machine with the hardware configuration listed above.  

The speedup is computed as:

$$S(p) = \frac{T_{seq}}{T_{p}}$$

The efficiency is computed as:

$$E(p) = \frac{S(p)}{p} \cdot 100\%$$

## 8. Conclusions

Both the sequential (SEQ) and MPI versions of the odd–even bubble sort were fully implemented.

## 9. References

- Гергель В.П. Высокопроизводительные вычисления для многопроцессорных многоядерных систем. Серия «Суперкомпьютерное образование».
- Гергель В.П. Теория и практика параллельных вычислений. – М.: Интуит Бином. Лаборатория знаний, 2007.
- Wilkinson B., Allen M. Parallel programming: techniques and applications using networked workstations and parallel computers. – Prentice Hall, 1999.

## Appendix

```cpp
bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(data_.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    if (rank != 0) {
      data_.assign(n, 0);
    }
    if (n > 0) {
      MPI_Bcast(data_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    }
    GetOutput() = data_;
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  ComputeScatterInfo(n, size, counts, displs);

  const int local_n = counts[rank];
  std::vector<int> local(local_n);

  MPI_Scatterv((rank == 0 ? data_.data() : nullptr),
               counts.data(), displs.data(), MPI_INT,
               local.data(), local_n, MPI_INT,
               0, MPI_COMM_WORLD);

  if (local_n > 1) {
    std::ranges::sort(local);
  }

  for (int phase = 0; phase < n; ++phase) {
    OddEvenExchange(local, counts, rank, size, phase);
  }

  std::vector<int> result;
  if (rank == 0) {
    result.resize(n);
  }

  MPI_Gatherv(local.data(), local_n, MPI_INT,
              (rank == 0 ? result.data() : nullptr),
              counts.data(), displs.data(), MPI_INT,
              0, MPI_COMM_WORLD);

  int out_n = 0;
  if (rank == 0) {
    out_n = n;
  }
  MPI_Bcast(&out_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = result;
  } else {
    GetOutput().assign(out_n, 0);
  }

  if (out_n > 0) {
    MPI_Bcast(GetOutput().data(), out_n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}
```
