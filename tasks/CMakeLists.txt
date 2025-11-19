project(parallel_programming_course LANGUAGES C CXX)

message(STATUS "Student's tasks")

# Test runner executables
set(FUNC_TEST_EXEC ppc_func_tests)
set(PERF_TEST_EXEC ppc_perf_tests)

# ——— Include helper scripts ——————————————————————————————————————
include(${CMAKE_SOURCE_DIR}/cmake/functions.cmake)

# ——— Initialize test executables —————————————————————————————————————
ppc_add_test(${FUNC_TEST_EXEC} common/runners/functional.cpp USE_FUNC_TESTS)
ppc_add_test(${PERF_TEST_EXEC} common/runners/performance.cpp USE_PERF_TESTS)

# ——— List of implementations ————————————————————————————————————————
set(PPC_IMPLEMENTATIONS "all;mpi;omp;seq;stl;tbb" CACHE STRING "Implementations to build (semicolon-separated)")

# ——— Configure each subproject —————————————————————————————————————
file(
  GLOB subdirs
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/*")

# Список папок для исключения чтобы не гонять экземпл тесты по двести лет
set(EXCLUDED_DIRS 
    "example_processes" 
    "example_processes_2" 
    "example_processes_3"
    "example_threads"
)

foreach(sub IN LISTS subdirs)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${sub}" AND 
     NOT sub STREQUAL "common" AND
     NOT sub IN_LIST EXCLUDED_DIRS)
    ppc_configure_subproject(${sub})
  endif()
endforeach()
